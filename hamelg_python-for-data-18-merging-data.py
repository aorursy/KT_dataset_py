import numpy as np

import pandas as pd

import os
table1 = pd.DataFrame({"P_ID" : (1,2,3,4,5,6,7,8),

                     "gender" : ("male", "male", "female","female",

                                "female", "male", "female", "male"),

                     "height" : (71,73,64,64,66,69,62,72),

                     "weight" : (175,225,130,125,165,160,115,250)})



table1
table2 = pd.DataFrame({"P_ID" : (1, 2, 4, 5, 7, 8, 9, 10),

                     "sex" : ("male", "male", "female","female",

                            "female", "male", "male", "female"),

                     "visits" : (1,2,4,12,2,2,1,1),

                     "checkup" : (1,1,1,1,1,1,0,0),

                     "follow_up" : (0,0,1,2,0,0,0,0),

                     "illness" : (0,0,2,7,1,1,0,0),

                     "surgery" : (0,0,0,2,0,0,0,0),

                     "ER" : ( 0,1,0,0,0,0,1,1) } ) 



table2
combined1 = pd.merge(table1,       # First table

                    table2,        # Second table

                    how="inner",   # Merge method

                    on="P_ID")     # Column(s) to join on



combined1
# A left join keeps all key values in the first(left) data frame



left_join = pd.merge(table1,       # First table

                    table2,        # Second table

                    how="left",   # Merge method

                    on="P_ID")     # Column(s) to join on



left_join
# A right join keeps all key values in the second(right) data frame



right_join = pd.merge(table1,       # First table

                    table2,        # Second table

                    how="right",   # Merge method

                    on="P_ID")     # Column(s) to join on



right_join
# An outer join keeps all key values in both data frames



outer_join = pd.merge(table1,      # First table

                    table2,        # Second table

                    how="outer",   # Merge method

                    on="P_ID")     # Column(s) to join on



outer_join
table1.rename(columns={"gender":"sex"}, inplace=True) # Rename "gender" column



combined2 = pd.merge(table1,               # First data frame

                  table2,                  # Second data frame

                  how="outer",             # Merge method

                  on=["P_ID","sex"])    # Column(s) to join on



combined2