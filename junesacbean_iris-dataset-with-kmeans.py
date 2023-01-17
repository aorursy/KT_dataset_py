from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score
import pandas as pd
from scipy import stats

train_data = pd.read_csv('train_data.csv')

test_data = pd.read_csv('test_data.csv')
train_y = pd.read_csv('train_y.csv')
train_data.head(3)
train_data.drop(columns = ['ID'], inplace = True)
kmeans = KMeans(n_clusters=3, random_state=0).fit(train_data) 

# 根據訓練資料及的眾數決定kernel核心
centers = kmeans.cluster_centers_
for i in range(3):
    p = kmeans.predict(train_data[train_y['class'] == i])
    # print(stats.mode(p))
    pp = stats.mode(p).mode[0] # 求實際類別為 i 所對應的類別標號 pp
    # print(pp.mode[0])
    kmeans.cluster_centers_[i] = centers[pp]
# 在訓練及上評估mapping是否正確
y_pred = kmeans.predict(train_data)

accuracy_score(train_y['class'], y_pred)
# 對測試資料進行預測
test_df = test_data.copy()
ID = test_df.pop('ID')

y_pred = kmeans.predict(test_df)
y_pred
# 製造submission dataframe
ans = pd.DataFrame(ID, columns = ['ID'])
ans['class'] = y_pred
ans
ans.to_csv('kmeans_pred.csv', index = False)
