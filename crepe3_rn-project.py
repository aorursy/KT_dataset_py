import pandas as pd
!ls '../input/'
data = pd.read_csv('../input/TRN', sep = '\t', index_col = 'INDEX')
data
ind1 = data['IND_BOM_1_1'] == 0 
ind2 = data['IND_BOM_1_2'] == 1
all(ind1 == ind2)
ind_cl1 = data['IND_BOM_1_1'] == 0
ind_cl2 = data['IND_BOM_1_1'] == 1
X_cl1 = data[ind_cl1]
X_cl2 = data[ind_cl2]
print(len(X_cl2))
print(len(X_cl1))
from imblearn.over_sampling import SMOTE 
use_cols = data.columns
use_cols = use_cols.drop(['IND_BOM_1_1', 'IND_BOM_1_2'])
X = data[use_cols]
y = data['IND_BOM_1_1']
ind_cl1 = y_pos == 0
ind_cl2 = y_pos == 1
X_cl1 = frame[ind_cl1]
X_cl2 = frame[ind_cl2]
print(len(X_cl2))
print(len(X_cl1))
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1/4, 
                                                  random_state=42, stratify=y)
X_train1, X_val1, y_train1, y_val1 = train_test_split(X_train, y_train, test_size=1/3, 
                                                  random_state=39, stratify=y_train)
X_train2, X_val2, y_train2, y_val2 = train_test_split(X_train, y_train, test_size=1/3, 
                                                  random_state=40, stratify=y_train)
X_train3, X_val3, y_train3, y_val3 = train_test_split(X_train, y_train, test_size=1/3, 
                                                  random_state=41, stratify=y_train)
X_train4, X_val4, y_train4, y_val4 = train_test_split(X_train, y_train, test_size=1/3, 
                                                  random_state=42, stratify=y_train)
X_train5, X_val5, y_train5, y_val5 = train_test_split(X_train, y_train, test_size=1/3, 
                                                  random_state=43, stratify=y_train)
X_test['y'] = y_test
X_test.to_csv('test.csv')
X_val1['y'] = y_val1
X_val1.to_csv('val1.csv')
X_val2['y'] = y_val2
X_val2.to_csv('val2.csv')
X_val3['y'] = y_val3
X_val3.to_csv('val3.csv')
!ls
X_val4['y'] = y_val4
X_val4.to_csv('val4.csv')
X_val5['y'] = y_val5
X_val5.to_csv('val5.csv')
sm = SMOTE()
X_pos1, y_pos1 = sm.fit_sample(X_train1, y_train1)
X_pos2, y_pos2 = sm.fit_sample(X_train2, y_train2)
X_pos3, y_pos3 = sm.fit_sample(X_train3, y_train3)
X_pos4, y_pos4 = sm.fit_sample(X_train4, y_train4)
X_pos5, y_pos5 = sm.fit_sample(X_train5, y_train5)
frame1 = pd.DataFrame(X_pos1)
frame1['y'] = y_pos1
frame1.to_csv('train1.csv')
frame2 = pd.DataFrame(X_pos2)
frame2['y'] = y_pos2
frame1.to_csv('train2.csv')
frame3 = pd.DataFrame(X_pos3)
frame3['y'] = y_pos3
frame1.to_csv('train3.csv')
frame4 = pd.DataFrame(X_pos4)
frame4['y'] = y_pos4
frame1.to_csv('train4.csv')
frame5 = pd.DataFrame(X_pos5)
frame5['y'] = y_pos5
frame1.to_csv('train5.csv')