#All without any preprocessing/normalisation tried
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
data_benign = pd.read_csv("../input/dm-assignment-3/train_benign.csv",sep=',')
#data_benign = pd.read_csv(r'C:\Users\91987\Documents\Fourth year\Data mining\New\Assignment 3\train_benign.csv', sep=',')
data_benign.head()
data_malware = pd.read_csv("../input/dm-assignment-3/train_malware.csv",sep=',')
#data_malware = pd.read_csv(r'C:\Users\91987\Documents\Fourth year\Data mining\New\Assignment 3\train_malware.csv', sep=',')
data_malware.head()
null_benign = data_benign.columns[data_benign.isnull().any()]
null_benign
null_malware = data_benign.columns[data_malware.isnull().any()]
null_malware
data_m = data_malware
data_b = data_benign
data_m['class']= 1
data_m.head()
data_b['class']=0
data_b.head()
data_class = pd.concat([data_b['class'],data_m['class']])
data_class.head()
data_class
final_data = pd.concat([data_benign,data_malware])
final_data.head()
final_data.tail()
final_data = final_data.drop(['class'], axis = 1)
final_data = final_data.drop(['FileName'],axis=1)
final_data.head()
diff = final_data.columns[final_data.dtypes == 'object']
diff
data_test = pd.read_csv("../input/dm-assignment-3/Test_data.csv")
#data_test = pd.read_csv(r'C:\Users\91987\Documents\Fourth year\Data mining\New\Assignment 3\Test_data.csv', sep=',')
data_test = data_test.drop(['FileName'],axis=1)
data_test = data_test.iloc[:, :-1]
data_test.head()
diff1 = data_test.columns[data_test.dtypes == 'object']
diff1
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(final_data, data_class, test_size=0.20 ,random_state=42)
np.random.seed(42)
from sklearn.ensemble import RandomForestClassifier

score_train_RF = []
score_test_RF = []

for i in range(5,20,1):
    rf = RandomForestClassifier(n_estimators = 1000, max_depth=i)
    rf.fit(X_train, y_train)
    sc_train = rf.score(X_train,y_train)
    score_train_RF.append(sc_train)
    sc_test = rf.score(X_test,y_test)
    score_test_RF.append(sc_test)

plt.figure(figsize=(10,6))
train_score,=plt.plot(range(5,20,1),score_train_RF,color='blue', linestyle='dashed', marker='o',markerfacecolor='green', markersize=5)
test_score,=plt.plot(range(5,20,1),score_test_RF,color='red',linestyle='dashed', marker='o', markerfacecolor='blue', markersize=5)
plt.legend( [train_score,test_score],["Train Score","Test Score"])
plt.title('Fig4. Score vs. No. of Trees')
plt.xlabel('No. of Trees')
plt.ylabel('Score')
rf = RandomForestClassifier(n_estimators=1000, max_depth = 18)
rf.fit(X_train, y_train)
rf.score(X_test,y_test)
check4 = rf.predict(data_test)
check4
index = []
for i in range(3370):
    index.append(int(i)+1)
df_ans = {'FileName':index}
df_ans = pd.DataFrame(df_ans)
df_ans['Class']= check4
df_ans.head()
from IPython.display import HTML
import pandas as pd
import numpy as np
import base64
def create_download_link(df, title = "Download CSV file", filename = "DataRF12.csv"):
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode())
    payload = b64.decode()
    html = '<a download="{filename}" href="data:text/csv;base64,{payload}" target="_blank">{title}</a>'
    html = html.format(payload=payload,title=title,filename=filename)
    return HTML(html)
create_download_link(df_ans)