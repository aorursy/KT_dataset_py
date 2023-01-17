import numpy as np

import pandas as pd

import statsmodels.api as sm

from statsmodels.formula.api import ols
# load data file

combine = pd.read_csv('../input/2019-nfl-scouting-combine/2019_nfl_combine_results.csv')
print(combine[0:3])
combine.rename(columns = {"Hand Size (in)": "hand_size", "Weight (lbs)":"weight", 'Height (in)':'height', 

                         "Arm Length (in)":"arm_length",'Vert Leap (in)':'vert','Broad Jump (in)':'broad', '40 Yard':'dash'}, inplace = True)

print(combine[0:3])
combine.POS.value_counts()
#Assign posititons to offense, defense, or special 

combine.loc[(combine['POS'] == "C") |(combine['POS'] == "FB") |(combine['POS'] == "OG") |(combine['POS'] == "OL") |

            (combine['POS'] == "OT") |(combine['POS'] == "QB") |(combine['POS'] == "RB") |(combine['POS'] == "TE") |

            (combine['POS'] == "WR") , "Side"] = "Offense"

combine.loc[(combine['POS'] == "CB") |(combine['POS'] == "DE") |(combine['POS'] == "DT") |(combine['POS'] == "EDG") |

            (combine['POS'] == "LB") |(combine['POS'] == "S"), "Side"] = "Defense"

combine.loc[(combine['POS'] == "K") |(combine['POS'] == "P") |(combine['POS'] == "LS"), "Side"] = "Special"
#Assign posititons to posistion zone

combine.loc[(combine['POS'] == "C") |(combine['POS'] == "OG") |(combine['POS'] == "OL") |

            (combine['POS'] == "OT") |(combine['POS'] == "DT") |(combine['POS'] == "DE") |(combine['POS'] == "EDG") 

            |(combine['POS'] == "TE") , "Zone"] = "Line"

combine.loc[(combine['POS'] == "LB") |(combine['POS'] == "RB")|(combine['POS'] == "FB"), "Zone"] = "Mid"

combine.loc[(combine['POS'] == "QB"), "Zone"] = "QB"

combine.loc[(combine['POS'] == "CB") |(combine['POS'] == "S")|(combine['POS'] == "WR"), "Zone"] = "Skilled"

combine.loc[(combine['POS'] == "K") |(combine['POS'] == "P") |(combine['POS'] == "LS"), "Zone"] = "Special"



print(combine[0:5])
combine.Zone.value_counts()
combine.Side.value_counts()
combine.College.value_counts()