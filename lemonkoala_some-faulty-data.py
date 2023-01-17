import pandas as pd
games       = pd.read_csv("../input/games.csv")
game_events = pd.read_csv("../input/game_events.csv")
games["player_count"].value_counts()
game_events[
     game_events["event"].isin(["upgrade:SH", "upgrade:TE"]) &
    (game_events["round"] == 1) &
    (game_events["turn"]  == 1) 
].head()
game_events[
    (game_events["game"]    == "0512") &
    (game_events["faction"] == "chaosmagicians") &
    (game_events["round"]    == 1)
]