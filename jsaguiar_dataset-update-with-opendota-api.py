from datetime import datetime
import time
import json
import requests
import numpy as np
import pandas as pd

class OpenDotaAPI():
    def __init__(self, verbose = False):
        self.verbose = verbose
        self.last_match_id = 0

    def _call(self, url, parameters, tries= 2):
        for i in range(tries):
            try:
                if self.verbose: print("Sending API request... ", end="", flush=True)
                resp = requests.get(url, params= parameters, timeout= 20)
                load_resp = json.loads(resp.text)
                if self.verbose: print("done")
                return load_resp
            except Exception as e:
                print("failed. Trying again in 5s")
                print(e)
                time.sleep(5)
        else:
            ValueError("Unable to connect to OpenDota API")

    # Return a list of 100 recent matches; save smaller match_id
    def get_recent_matches(self, use_last_match = False):
        params = dict()
        if use_last_match:
            params['less_than_match_id'] = self.last_match_id
        url = "https://api.opendota.com/api/publicMatches"
        matches = self._call(url, params)
        self.last_match_id = min([item['match_id'] for item in matches])
        return matches

    # Return a dictionary with match information
    def get_match_info(self, match_id):
        url = "https://api.opendota.com/api/matches/" + str(match_id)
        return self._call(url, None)

    # Return a list with player's match history (previous matches)
    def get_player_matches_history(self, account_id):
        url = "https://api.opendota.com/api/players/{}/matches".format(account_id)
        return self._call(url, None)

    # Get a dictionary with overall benchmarks given account id (kills, deaths, gpm...)
    def get_player_totals(self, account_id, hero_id = None):
        params = {'sort': 1}
        if hero_id: params['hero_id'] = hero_id
        url = "https://api.opendota.com/api/players/{}/totals".format(int(account_id))
        return self._call(url, params)

    # Return wins and losses for a given account id
    def get_player_win_loss(self, account_id, hero_id = None):
        if hero_id:
            params = {'hero_id': hero_id}
        else:
            params = None
        url = "https://api.opendota.com/api/players/{}/wl".format(account_id)
        resp = self._call(url, params)
        return resp['win'], resp['lose']
class DataPreprocessing():
    def __init__(self):
        # Initialize tables as empty dataframes
        self.matches = pd.DataFrame()
        self.players = pd.DataFrame()
        self.chat = pd.DataFrame()
        self.objectives = pd.DataFrame()
        self.advantages = pd.DataFrame()
        self.events = pd.DataFrame()
        self.abilities = pd.DataFrame()
        self.wards = pd.DataFrame()
        self.previous_matches = pd.DataFrame()


    def get_match(self, match):
        """ Get general information from the match and append to self.matches. """
        
        fields = ['match_id', 'match_seq_num', 'patch', 'region', 'start_time', 'duration',
            'game_mode', 'skill', 'first_blood_time', 'barracks_status_dire',
            'barracks_status_radiant', 'tower_status_dire', 'tower_status_radiant',
            'dire_score', 'radiant_score', 'radiant_win']

        proc_match = {key: [match[key]] for key in fields}
        self.matches = self.matches.append(pd.DataFrame(proc_match), ignore_index= True)
    
    def get_match_chat(self, match):
        """ Get match chat and save to self.chat dataframe. """
        fields = ['match_id', 'time', 'type', 'key', 'slot', 'player_slot']
        messages = []
        if match['chat']:
            for item in match['chat']:
                message = {'match_id': match['match_id']}
                for field in fields:
                    message[field] = item[field]
                messages.append(message.copy())
            if messages:
                self.chat = self.chat.append(pd.DataFrame(messages), ignore_index= True)

    def get_match_objectives(self, match):
        """ Get game objectives like Roshan and towers and append to self.objectives dataframe. """
        fields = ['time', 'type', 'unit', 'key', 'slot', 'player_slot']
        objectives = []
        if match['objectives']:
            for item in match['objectives']:
                obj = {'match_id': match['match_id']}
                for field in fields:
                    if field in item:
                        obj[field] = item[field]
                    else:
                        obj[field] = np.nan
                objectives.append(obj.copy())
        if objectives:
            self.objectives = self.objectives.append(pd.DataFrame(objectives), ignore_index= True)

    def get_match_advantages(self, match):
        """ Get radiant gold and xp advantage for each minute and append to self.advantages dataframe. """
        advantages = []
        if match['radiant_gold_adv']:  # Gold advantage (gold_or_xp = 0)
            for i, value in enumerate(match['radiant_gold_adv']):
                adv = {
                    'match_id': match['match_id'],
                    'minute': i,
                    'gold_or_xp': 0,
                    'value': int(value)
                }
                advantages.append(adv.copy())
        if match['radiant_xp_adv']:  # XP advantage (gold_or_xp = 1)
            for i, value in enumerate(match['radiant_xp_adv']):
                adv = {
                    'match_id': match['match_id'],
                    'minute': i,
                    'gold_or_xp': 1,
                    'value': int(value)
                }
                advantages.append(adv.copy())
        if advantages:
            self.advantages = self.advantages.append(pd.DataFrame(advantages), ignore_index= True)

    def get_players_events(self, match):
        """ Get events for each player (kills, runes, bb and purchases) and append to self.events. """
        events = []
        for player in match['players']:
            if player['buyback_log']: # Player's Buybacks
                for bb in player['buyback_log']:
                    tmp = {
                        'match_id': match['match_id'],
                        'account_id': player['account_id'],
                        'player_slot': player['player_slot'],
                        'hero_id': player['hero_id'],
                        'time': bb['time'],
                        'key': np.nan,
                        'event': 'buyback'
                    }
                    events.append(tmp.copy())
            if player['kills_log']: # Player's kills on enemy heroes
                for kill in player['kills_log']:
                    tmp = {
                        'match_id': match['match_id'],
                        'account_id': player['account_id'],
                        'player_slot': player['player_slot'],
                        'hero_id': player['hero_id'],
                        'time': kill['time'],
                        'key': kill['key'],
                        'event': 'kill'
                    }
                    events.append(tmp.copy())
            if player['runes_log']: # Runes picked
                for rune in player['runes_log']:
                    tmp = {
                        'match_id': match['match_id'], 
                        'account_id': player['account_id'], 
                        'player_slot': player['player_slot'],
                        'hero_id': player['hero_id'], 
                        'time': rune['time'],
                        'key': rune['key'],
                        'event': 'rune'
                    }
                    events.append(tmp.copy())
            if player['purchase_log']:
                for item in player['purchase_log']: # Items purchased
                    tmp = {
                        'match_id': match['match_id'],
                        'account_id': player['account_id'],
                        'player_slot': player['player_slot'],
                        'hero_id': player['hero_id'],
                        'time': item['time'],
                        'key': item['key'],
                        'event': 'purchase'
                    }
                    events.append(tmp.copy())
        if events:
            self.events = self.events.append(pd.DataFrame(events), ignore_index= True)

    def get_ability_upgrades(self, match):
        """ Get skill upgrades for each player. Columns goes from 1 to 25 for each possible skill upgrade. """
        ability_upgrades = []
        for player in match['players']:
            if player['ability_upgrades_arr']:
                tmp = {
                    'match_id': match['match_id'],
                    'account_id': player['account_id'],
                    'player_slot': player['player_slot'],
                    'hero_id': player['hero_id'],
                }
                for i in range(25):
                    tmp['skill_upgrade_' + str(i + 1)] = np.nan
                for i, value in enumerate(player['ability_upgrades_arr']):
                    tmp['skill_upgrade_' + str(i + 1)] = value
                ability_upgrades.append(tmp.copy())
        if ability_upgrades:
            self.abilities = self.abilities.append(pd.DataFrame(ability_upgrades), ignore_index= True)

    def get_wards(self, match):
        """ Get time, position, slot and hero for each ward placed and append to self.wards dataframe. """
        wards = []
        for player in match['players']:
            if player['obs_log']:  # Observer wards (type = 0)
                for item in player['obs_log']:
                    ward = {
                        'match_id': match['match_id'], 'account_id': player['account_id'],
                        'player_slot': player['player_slot'], 'hero_id': player['hero_id'],
                        'time': item['time'], 'x': item['x'], 'y': item['y'], 'type': 0
                    }
                    wards.append(ward.copy())
            if player['sen_log']:  # Sentry wards (type = 1)
                for item in player['sen_log']:
                    ward = {
                        'match_id': match['match_id'], 'account_id': player['account_id'],
                        'player_slot': player['player_slot'], 'hero_id': player['hero_id'],
                        'time': item['time'], 'x': item['x'], 'y': item['y'], 'type': 1
                    }
                    wards.append(ward.copy())
        if wards:
            self.wards = self.wards.append(pd.DataFrame(wards), ignore_index= True)

    def get_players(self, match):
        """ Get match information for each player and append to self.players dataframe. """
        
        fields = ['player_slot', 'account_id', 'hero_id', 'kills', 'deaths',
            'assists', 'last_hits', 'denies', 'gold_per_min', 'xp_per_min',
            'gold_spent', 'hero_damage', 'hero_healing', 'tower_damage',
            'level', 'party_size', 'item_0', 'item_1', 'item_2', 'item_3',
            'item_4', 'item_5', 'camps_stacked', 'creeps_stacked', 'obs_placed', 'sen_placed',
            'purchase_tpscroll', 'rune_pickups', 'roshans_killed', 'towers_killed', 'win']

        players = []
        for item in match['players']:
            player = {'match_id': match['match_id']}
            for field in fields:
                if field in item:
                    player[field] = item[field]
                else:
                    player[field] = np.nan
            players.append(player.copy())
        if players:
            self.players = self.players.append(pd.DataFrame(players), ignore_index= True)

    def get_previous_matches(self, current_match_id, player_account_id, player_previous_matches,
                             current_match_start_time):
        """ Append all previous matches before match_start_time from a given account id. """
        
        previous_matches = []
        fields = ['match_id', 'player_slot', 'radiant_win', 'duration', 'game_mode',
                  'lobby_type', 'start_time', 'version', 'hero_id', 'kills', 'deaths',
                  'assists', 'skill', 'leaver_status', 'party_size']

        for item in player_previous_matches:
            previous_match = {'current_match_id': current_match_id, 'account_id': player_account_id}
            for field in fields:
                previous_match[field] = item[field]
            previous_matches.append(previous_match.copy())

        df = pd.DataFrame(previous_matches)
        # Avoid future games
        df = df[df['start_time'] < current_match_start_time]
        self.previous_matches = self.previous_matches.append(df, ignore_index= True)

    def get_all_current_match_tables(self, match_details):
        """ Get all tables from a current match, except the previous matches. """
        self.get_match(match_details)
        self.get_players(match_details)
        self.get_match_chat(match_details)
        self.get_match_objectives(match_details)
        self.get_match_advantages(match_details)
        self.get_ability_upgrades(match_details)
        self.get_players_events(match_details)
        self.get_wards(match_details)
def main(sleep_time = 2):
    api = OpenDotaAPI(verbose= True)
    data = DataPreprocessing()
    recent_matches = filter_matches(api.get_recent_matches())
    for recent_match in recent_matches:
        time.sleep(sleep_time)
        match_details = api.get_match_info(recent_match['match_id'])
        data.get_all_current_match_tables(match_details)
        # Get previous matches for all players with valid account ids
        players_with_account = data.players[data.players['account_id'] > 0]
        for i, player in players_with_account.iterrows():
            time.sleep(sleep_time)
            full_match_history = api.get_player_matches_history(player['account_id'])
            if full_match_history:
                data.get_previous_matches(match_details['match_id'], player['account_id'],
                full_match_history, match_details['start_time'])
    return data


def filter_matches(matches_list):
    return list(filter(lambda m: _filter_function(m), matches_list))

def _filter_function(match):
    if match['duration'] < 1000 or match['duration'] > 4200:
        return False
    elif match['lobby_type'] < 5 or match['lobby_type'] > 7:
        return False
    else:
        return True
def append_sqlite(data):
    data.matches.to_sql(if_exists = 'append')