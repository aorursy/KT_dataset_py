import pandas as pd
import numpy as np
from pandas import DataFrame
import matplotlib.pyplot as plt
import os
print(os.listdir("../input"))
multiple = pd.read_csv('../input/multipleChoiceResponses.csv', dtype=np.object)

ne = multiple[multiple.Q6 == "Not employed"]
q4_ne = ne[["Q4"]]
q4_nep = q4_ne.Q4.value_counts().plot(kind="pie", autopct='%1.1f%%',figsize = (10,10))


q2_ne = ne[["Q2"]]
q2_nep = q2_ne.Q2.value_counts().plot(kind="pie",autopct='%1.1f%%',figsize = (10,10))
q3_ne = ne[["Q3"]]
q3_ne.Q3.value_counts().plot(kind="pie",autopct='%1.1f%%',figsize = (10,10))

st = multiple[multiple.Q6 == "Student"]
q1_st = st[["Q2"]]
q1_st.Q2.value_counts().plot(kind="pie",label=False,title = "Age of students who answered the kaggle's survey" ,autopct='%1.1f%%',figsize = (10,10))
plt.legend()
plt.show()
q3_st = st[["Q3"]]
q3_st.Q3.value_counts().plot(kind="pie",title= "", autopct='%1.1f%%',figsize = (10,10))
q12_st = st[["Q12_MULTIPLE_CHOICE"]]
q12_st.Q12_MULTIPLE_CHOICE.value_counts().plot(kind="pie",title= "", autopct='%1.1f%%',figsize = (10,10))
q17_st = st[["Q17"]]
q17_st.Q17.value_counts().plot(kind="pie",title= "the specific programming lenguage used most often", autopct='%1.1f%%',figsize = (10,10))
q23_st = st[["Q23"]]
q23_st.Q23.value_counts().plot(kind="pie",title= "Percent of the time that students spent actively coding", autopct='%1.1f%%',figsize = (10,10))
q26_st = st[["Q26"]]
q26_st.Q26.value_counts().plot(kind="pie",title= "", autopct='%1.1f%%',figsize = (10,10))

