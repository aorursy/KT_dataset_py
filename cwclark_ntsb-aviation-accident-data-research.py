import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt

#from subprocess import check_output
#print(check_output(["ls", "../input"]).decode("utf8"))

accidents = pd.read_csv("../input/aircraft.txt", encoding="iso-8859-1", dtype={4:np.str,7:np.str,17:np.str,32:np.str,53:np.str,72:np.str,73:np.str,76:np.str,82:np.str})
accidents = accidents.dropna(axis=1, how="all")
injuries = pd.read_csv("../input/injury.txt")

injuries.info()
accidents.info()
#Combine injuries df with accidents df
def combine_inj_acc(acc_df, inj_df):
    print(inj_df["inj_person_category"].value_counts())
    print(inj_df["injury_level"].value_counts())
    print(inj_df["inj_person_count"].value_counts())
    
combine_inj_acc(accidents, injuries)
