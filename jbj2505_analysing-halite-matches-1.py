! pip install kaggle_environments==0.2.1 mypy
from typing import List
import json
from kaggle_environments import make
from kaggle_environments.envs.halite.halite import get_to_pos
from kaggle_environments.utils import Struct
import pandas as pd
env = make("halite")
_ = env.reset(num_agents=4)
def convert_to_struct(obj):
    """
    Converts an object to the Kaggle `Struct` format.
    """
    if isinstance(obj, list):
        return [convert_to_struct(item) for item in obj]
    if isinstance(obj, dict):
        return Struct(**{key: convert_to_struct(value) for key, value in obj.items()})
    return obj


with open("../input/halite-match-steps/steps.json", mode="r") as file_pointer:
    env.steps = convert_to_struct(json.load(file_pointer))
env.render(mode="ipython", width=800, height=600)
def make_actions_df(steps: List[List[Struct]]) -> pd.DataFrame:
    
    def actions_from_steps(steps):
        for step, state in enumerate(steps):
            step = state[0].observation.step
            for player_index, player in enumerate(state):
                for uid, action in player.action.items():
                    yield {"step": step, "uid": uid, "action": action}
                    
    return pd.DataFrame(actions_from_steps(steps))


def make_ships_df(steps: List[List[Struct]]) -> pd.DataFrame:
    
    def ships_from_steps(steps):
        for step, state in enumerate(steps):
            step = state[0].observation.step
            for player_index, player in enumerate(state[0].observation.players):
                for uid, (pos, halite) in player[2].items():
                    yield {"step": step, "uid": uid, "pos": pos, "halite": halite, "player": player_index}
                    
    return pd.DataFrame(ships_from_steps(steps))

def make_shipyards_df(steps: List[List[Struct]]) -> pd.DataFrame:
    
    def shipyard_from_state(state):
        for step, state in enumerate(steps):
            step = state[0].observation.step
            for player_index, player in enumerate(state[0].observation.players):
                for uid, pos in player[1].items():
                    yield {"step": step, "uid": uid, "pos": pos, "player": player_index}
                
    return pd.DataFrame(shipyard_from_state(steps[-1]))

def make_players_df(steps: List[List[Struct]]) -> pd.DataFrame:
    
    def players_from_steps(steps):
        for step, state in enumerate(steps):
            step
            for player_index, player in enumerate(state[0].observation.players):
                yield {"step": step, "player": player_index, "halite": player[0]}
                
    return pd.DataFrame(players_from_steps(steps))
actions_df = make_actions_df(env.steps)
actions_df.head()
ships_df = make_ships_df(env.steps)
ships_df.head()
shipyards_df = make_shipyards_df(env.steps)
shipyards_df.head()
players_df = make_players_df(env.steps)
players_df.head()
# Some pipeline functions for our advanced tables.

def add_halite_delta(df: pd.DataFrame) -> pd.DataFrame:
    def _halite_delta(ship):
        ship = ship.sort_values("step", ascending=True)
        return ship["halite"] - ship.shift()["halite"]
    df["halite_delta"] = df.groupby("uid").apply(_halite_delta).reset_index("uid")["halite"]
    return df
    
def add_mine_deposit_actions(df: pd.DataFrame) -> pd.DataFrame:
    shipyard_present = ~pd.isna(
        df
        .merge(shipyards_df, how="left", on=["step", "pos"], suffixes=["_ship", "_shipyard"])
        ["uid_shipyard"]
    )
    
    filter_ = (pd.isna(df["action"])) & (~pd.isna(df["halite_delta"]))
    
    df.loc[filter_ & shipyard_present, "action"] = "DEPOSIT"
    df.loc[filter_ & (~shipyard_present), "action"] = "MINE"

    return df

def add_halite_delta_abs(df: pd.DataFrame) -> pd.DataFrame:
    df["halite_delta_abs"] = df["halite_delta"].abs()
    return df

def add_step_prev(df: pd.DataFrame) -> pd.DataFrame:
    df["step_prev"] = df["step"] - 1
    return df

def add_expected_pos(df: pd.DataFrame) -> pd.DataFrame:
    df["expected_pos"] = df.apply(lambda ship: get_to_pos(env.configuration.size, ship["pos_prev"], ship["action"]), axis=1)
    return df
ship_actions_df = (
    actions_df
    .copy()
    .pipe(lambda df: df[df["action"].isin(("NORTH", "SOUTH", "EAST", "WEST", "CONVERT"))])
    .merge(ships_df, how="outer", on=["step", "uid"])
    .pipe(add_halite_delta)
    .pipe(add_mine_deposit_actions)
)
ship_actions_df.head()
# Number of actions sent per action type.
(
    ship_actions_df
    .groupby("action")
    .size()
    .sort_values(ascending=False)
    .plot(kind="bar")
)
# Amount of halite minded (deducting the halite spent on moving) per ship.
(
    ship_actions_df
    [ship_actions_df["action"] != "DEPOSIT"]
    .groupby("uid")
    ["halite_delta"]
    .sum()
    .sort_values(ascending=False)
    .head(5)
    .plot(kind="bar")
)
shipyard_actions_df = (
    actions_df
    .copy()
    .pipe(lambda df: df[df["action"].isin(("SPAWN", ))])
    .merge(shipyards_df, how="right", on=["step", "uid"])
)
shipyard_actions_df.head()
# Number of spawn actions by shipyard.
(
    shipyard_actions_df
    [shipyard_actions_df["action"] == "SPAWN"]
    .groupby("uid")
    .size()
    .sort_values(ascending=False)
    .head()
    .plot(kind="bar")
)
deposit_df = (
    ship_actions_df
    [(ship_actions_df["action"] == "DEPOSIT") & (~pd.isna(ship_actions_df["halite_delta"]))]
    .merge(shipyards_df, how="left", on=["step", "pos"], suffixes=["_ship", "_shipyard"])
    .pipe(add_halite_delta_abs)
    [["step", "pos", "uid_ship", "uid_shipyard", "player_ship", "halite_delta_abs"]]
    .rename({"player_ship": "player", "halite_delta_abs": "halite"}, axis=1)
)
deposit_df.head()
(
    deposit_df
    .groupby("uid_shipyard")
    ["halite"]
    .sum()
    .sort_values(ascending=False)
    .head()
    .plot(kind="bar")
)
ship_collision_df = (
    ship_actions_df
    .groupby("uid")
    .apply(lambda ship: ship.sort_values("step").tail(1))
    .reset_index(drop=True)
    .pipe(add_step_prev)
    .merge(ships_df, how="left", left_on=["uid", "step_prev"], right_on=["uid", "step"], suffixes=["", "_prev"])
    .pipe(add_expected_pos)
    [["step", "uid", "expected_pos"]]
    .rename({"expected_pos": "pos"}, axis=1)
    .append(ships_df[["step", "uid", "pos"]])
    .groupby(["step", "pos"])["uid"].aggregate(lambda x: set(x)).reset_index()
    .pipe(lambda df: df[df["uid"].apply(lambda x: len(x) > 1)])
)

ship_collision_df