import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import glob

import os



for f in glob.glob('../input/*'):

    print(f, round(os.path.getsize(f) / 1000000, 2), 'MB')
explanations = {

    "A_CRAINJ": "Crash Injury Type",

    "A_CT": "Crash Type",

    "A_D15_19": "Involving a Young Driver (Aged 15-19)",

    "A_D15_20": "Involving a Young Driver (Aged 15-20)",

    "A_D16_19": "Involving a Young Driver (Aged 16-19)",

    "A_D16_20": "Involving a Young Driver (Aged 16-20)",

    "A_D16_24": "Involving a Young Driver (Aged 16-24)",

    "A_D21_24": "Involving a Young Driver (Aged 21-24)",

    "A_D65PLS": "Involving an Older Driver (Aged 65+)",

    "A_DIST": "Involving a Distracted Driver",

    "A_DOW": "Day of Week",

    "A_DROWSY": "Involving a Drowsy Driver",

    "A_HR": "Involving a Hit and Run",

    "A_INTER": "Interstate",

    "A_INTSEC": "Intersection",

    "A_JUNC": "Junction",

    "A_LT": "Involving a Large Truck",

    "A_MANCOL": "Manner of Collision",

    "A_MC": "Involving a Motorcycle",

    "A_PED": "Involving a Pedestrian",

    "A_PEDAL": "Involving a Pedalcyclist",

    "A_POLPUR": "Involving a Police Pursuit",

    "A_POSBAC": "Involving a Driver With Positive BAC",

    "A_RD": "Involving a Roadway Departure",

    "A_REGION": "NHTSA Region",

    "A_RELRD": "Relationship to the Road",

    "A_ROADFC": "Roadway Function Class",

    "A_ROLL": "Involving a Rollover",

    "A_RU": "Rural/Urban",

    "A_SPCRA": "Involving Speeding",

    "A_TOD": "Time of Day"

}



acc = pd.read_csv('../input/ACC_AUX.CSV')

for c in acc.columns:

    if c in explanations:

        print('Explanation:', explanations[c])

    print(acc[c].describe())

    print('\n')
explanations = {

    "A_BODY": "Vehicle Body Type",

    "A_CDL_S": "CDL Status",

    "A_DRDIS": "Distracted Driver",

    "A_DRDRO": "Drowsy Driver",

    "A_IMP1": "Intial Impact Point",

    "A_IMP2": "Principal Impact Point",

    "A_LIC_C": "License Compliance",

    "A_LIC_S": "License Status",

    "A_MC_L_S": "Motorcycle License Status",

    "A_SBUS": "School Bus",

    "A_SPVEH": "Speeding Vehicle",

    "A_VROLL": "Rollover"

    }



veh = pd.read_csv('../input/VEH_AUX.CSV')

for c in veh.columns:

    if c in explanations:

        print('Explanation:', explanations[c])

    print(veh[c].describe())

    print('\n')
explanations = {

    "A_AGE1": "AGE Group 1",

    "A_AGE2": "AGE Group 2",

    "A_AGE3": "AGE Group 3",

    "A_AGE4": "AGE Group 4",

    "A_AGE5": "AGE Group 5",

    "A_AGE6": "AGE Group 6",

    "A_AGE7": "AGE Group 7",

    "A_AGE8": "AGE Group 8",

    "A_AGE9": "AGE Group 9",

    "A_ALCTES": "Alcohol Testing",

    "A_EJECT": "Ejection",

    "A_HISP": "Hispanic Origin",

    "A_HRACE": "Race and Hispanic - Using OMB Guidelines",

    "A_LOC": "Non-Motorist Location",

    "A_PERINJ": "Person Injury Type",

    "A_PTYPE": "Person Type",

    "A_RCAT": "Race - Using OMB Guidelines",

    "A_REST": "Restraint Use"

    }



per = pd.read_csv('../input/PER_AUX.CSV')

for c in per.columns:

    if c in explanations:

        print('Explanation:', explanations[c])

    print(per[c].describe())

    print('\n')