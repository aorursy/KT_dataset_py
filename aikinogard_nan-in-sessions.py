import numpy as np
import pandas as pd
sessions = pd.read_csv("../input/sessions.csv")
print(sessions.apply(lambda x: x.nunique(),axis=0))
print ("NaN percentage in action:", np.sum(sessions.action.isnull()) / len(sessions.action))
sessions[sessions.action.isnull()].action_type.value_counts()
print ("NaN percentage in action_type:", np.sum(sessions.action_type.isnull()) / len(sessions.action_type))
