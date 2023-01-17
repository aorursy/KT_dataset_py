import pandas as pd
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
file_path = '/kaggle/input/data-police-shootings/fatal-police-shootings-data.csv'
shootings = pd.read_csv(file_path, index_col = 'id')
shootings.head()
shootings.isnull().sum()
shootings = shootings.loc[shootings.race.notnull()]
shootings.isnull().sum()
(round(shootings.isna().sum()/shootings.shape[0]*100, 2)).astype('str') + '%'
shootings = shootings.dropna()
print(shootings.manner_of_death.unique())
print(shootings.armed.unique())
print(shootings.signs_of_mental_illness.unique())
print(shootings.threat_level.unique())
print(shootings.flee.unique())
print(shootings.body_camera.unique())
unknowns_dict = ['unknown', 'undetermined', 'unidentifiable', 'claimed to be armed']
shootings.armed.loc[shootings.armed.str.contains('|'.join(unknowns_dict),case = False)] = 'unknown'
dictionary = {
    'Guns': 
        ['bb gun', 'pellet gun', 'air pistol', 'bean-bag gun', 'gun'],
    'Blunt instruments':
        ['hammer', 'axe', 'ax', 'hatchet', 'crowbar', 'pole', 'rod', 'walking stick', 'stick', 'rock', 'baton', 'shovel',
        'metal object', 'baseball bat', 'bat', 'flagpole', 'metal pole', 'metal stick', 'blunt object', 'metal pipe', 
         'carjack', 'brick', 'garden tool', 'metal rake', 'mace', 'wrench', 'pipe'],
    'Sharp objects':
        ['knife', 'bayonet', 'razor', 'blade', 'machete', 'sword', 'chainsaw', 'chain saw' , 'sharp object', 'scissor',
         'scissors','chain saw', 'glass shard', 'samurai sword', 'lawn mower blade', 'box-cutter', 'straight edge razor',
        'beer bottle', 'bottle', 'sharp object','meat cleaver', 'box cutter'],
    'Piercing objects':
        ['spear', 'pick-axe', 'pick axe', 'pitchfork', 'cordless drill', 'nail gun', 'nailgun', 'pen',
        'crossbow', 'arrow and bow', 'screwdriver', 'ice pick', 'bow and arrow'],
    'Other unusual objects':
        ['oar', 'chair', 'barstool', 'pepper spray', 'spray', 'wasp spray', 'piece of wood',
        'toy weapon', 'torch', 'flashlight', 'air conditioner', 'hand torch', "contractor's level", 'chain', 'stapler'],
    'Hand tools':
        ['metal hand tool'],
    'Vehicles':
        ['motorcycle', 'car', 'van', 'wagon', 'bike', 'vehicle'],
    'Electrical devices':
        ['taser'],
    'Explosives':
        ['fireworks', 'grenade', 'molotov cocktail', 'incendiary device']
}
def categorize_arms(row):
    armed = row['armed'].lower()
    if 'and' in armed.split(' '): #using 'and' as separator to recognize multiple weapons
        row['arms_category'] = 'Multiple'
    elif armed in dictionary.keys():
        row['arms_category'] = armed
    elif armed == 'unknown':
        row['arms_category'] = 'Unknown'
    elif armed == 'unarmed':
        row['arms_category'] = 'Unarmed'
    else:
        for key, value in dictionary.items():
            if armed in value:
                row['arms_category'] = key
    return row

arms_categorized = shootings
arms_categorized['arms_category'] = None
arms_categorized = arms_categorized.apply(categorize_arms, axis = 'columns')
shootings = arms_categorized
shootings.head()
def assign_Race(race):
    if race == 'A':
        race = 'Asian'
    elif race == 'B':
        race = 'Black'
    elif race == 'H':
        race = 'Hispanic'
    elif race == 'N':
        race = 'Native'
    elif race == 'O':
        race = 'Other'
    elif race == 'W':
        race = 'White'
    return race
shootings.loc[:,'race'] = shootings.race.apply(assign_Race)