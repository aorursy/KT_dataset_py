import numpy as np

import pandas as pd
titanic_train = pd.read_csv("../input/train.csv")      # Read the data



char_cabin = titanic_train["Cabin"].astype(str)    # Convert cabin to str



new_Cabin = np.array([cabin[0] for cabin in char_cabin]) # Take first letter



titanic_train["Cabin"] = pd.Categorical(new_Cabin)  # Save the new cabin var
my_tab = pd.crosstab(index=titanic_train["Survived"],  # Make a crosstab

                     columns="count")                  # Name the count column



my_tab
type(my_tab)             # Confirm that the crosstab is a DataFrame
pd.crosstab(index=titanic_train["Pclass"],  # Make a crosstab

            columns="count")                # Name the count column
pd.crosstab(index=titanic_train["Sex"],     # Make a crosstab

                      columns="count")      # Name the count column
cabin_tab = pd.crosstab(index=titanic_train["Cabin"],  # Make a crosstab

                        columns="count")               # Name the count column



cabin_tab 
titanic_train.Sex.value_counts()
print (cabin_tab.sum(), "\n")   # Sum the counts



print (cabin_tab.shape, "\n")   # Check number of rows and cols



cabin_tab.iloc[1:7]             # Slice rows 1-6
cabin_tab/cabin_tab.sum()
# Table of survival vs. sex

survived_sex = pd.crosstab(index=titanic_train["Survived"], 

                           columns=titanic_train["Sex"])



survived_sex.index= ["died","survived"]



survived_sex
# Table of survival vs passenger class

survived_class = pd.crosstab(index=titanic_train["Survived"], 

                            columns=titanic_train["Pclass"])



survived_class.columns = ["class1","class2","class3"]

survived_class.index= ["died","survived"]



survived_class
# Table of survival vs passenger class

survived_class = pd.crosstab(index=titanic_train["Survived"], 

                            columns=titanic_train["Pclass"],

                             margins=True)   # Include row and column totals



survived_class.columns = ["class1","class2","class3","rowtotal"]

survived_class.index= ["died","survived","coltotal"]



survived_class
survived_class/survived_class.loc["coltotal","rowtotal"]
survived_class/survived_class.loc["coltotal"]
survived_class.div(survived_class["rowtotal"],

                   axis=0)
survived_class.T/survived_class["rowtotal"]
surv_sex_class = pd.crosstab(index=titanic_train["Survived"], 

                             columns=[titanic_train["Pclass"],

                                      titanic_train["Sex"]],

                             margins=True)   # Include row and column totals



surv_sex_class
surv_sex_class[2]        # Get the subtable under Pclass 2
surv_sex_class[2]["female"]   # Get female column within Pclass 2
surv_sex_class/surv_sex_class.loc["All"]    # Divide by column totals