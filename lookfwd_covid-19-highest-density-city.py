import pandas as pd



population_density = pd.read_csv('../input/covid19highestcitypopulationdensity/population_density.csv')



density_map = {(i, j): k for i, j, k in population_density[["Country/Region", "Province/State", "density"]].values}

dencity_map = {(i, j): k for i, j, k in population_density[["Country/Region", "Province/State", "mostdensecity"]].values}



population_density
def to_key(row):

    return (row['Country/Region'], row['Province/State'])



if False:

    train["density"] = train.apply(lambda row: density_map[to_key(row)], axis=1)

    train["densecity"] = train.apply(lambda row: dencity_map[to_key(row)], axis=1)