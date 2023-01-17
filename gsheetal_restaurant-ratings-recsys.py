from surprise import Dataset
from surprise import Reader
import pandas as pd
import numpy as np
import matplotlib.pyplot as pic
import os
#load the data
file_path = os.path.expanduser('..../restaurant_ratings.txt')
reader = Reader(line_format='user item rating timestamp', sep='\t')
data = Dataset.load_from_file(file_path, reader=reader)
#Q5
from surprise import SVD
from surprise.model_selection import cross_validate

algorithm = SVD()
performance = cross_validate(algorithm, data, measures=['RMSE', 'MAE'], cv=3, verbose=True)
print("Performance:",performance)
#Q6
algorithm = SVD(biased=False)
performance = cross_validate(algorithm, data, measures=['RMSE', 'MAE'], cv=3, verbose=True)
print("Performance:",performance)
#Q7
from surprise import NMF
algorithm = NMF()
performance = cross_validate(algorithm, data, measures=['RMSE', 'MAE'], cv=3, verbose=True)
print("Performance:",performance)
#Q8
from surprise import KNNBasic

algorithm = KNNBasic(sim_options = {
'user_based': True
})
performance = cross_validate(algorithm, data, measures=['RMSE', 'MAE'], cv=3, verbose=True)
print("Performance:",performance)
#Q9
algorithm = KNNBasic(sim_options = {
'user_based': False
})
performance = cross_validate(algorithm, data, measures=['RMSE', 'MAE'], cv=3, verbose=True)
print("Performance:",performance)
#Q14.a.1

import random
import numpy as np
set_seed = 10
random.seed(set_seed)
np.random.seed(set_seed)

algorithm = KNNBasic(sim_options = {
'name':'MSD',
'user_based': True
})
name_user_01 = "MSD"
per_user_01 = cross_validate(algorithm, data, measures=['RMSE', 'MAE'], cv=3, verbose=True)
print("Performance:",per_user_01)
#Q14.a.2
set_seed = 11
random.seed(set_seed)
np.random.seed(set_seed)

algorithm = KNNBasic(sim_options = {
'name':'cosine',
'user_based': True
})
name_user_02 = "cosine"
per_user_02 = cross_validate(algorithm, data, measures=['RMSE', 'MAE'], cv=3, verbose=True)
print("Performance:",per_user_02)
#Q14.a.3
set_seed = 12
random.seed(set_seed)
np.random.seed(set_seed)

algorithm = KNNBasic(sim_options = {
'name':'pearson',
'user_based': True
})
name_user_03 = "pearson"
per_user_03 = cross_validate(algorithm, data, measures=['RMSE', 'MAE'], cv=3, verbose=True)
print("Performance:",per_user_03)
#Q14.b.1
set_seed = 13
random.seed(set_seed)
np.random.seed(set_seed)

algorithm = KNNBasic(sim_options = {
'name':'MSD',
'user_based': False
})
name_item_01 = "MSD"
per_item_01 = cross_validate(algorithm, data, measures=['RMSE', 'MAE'], cv=3, verbose=True)
print("Performance:",per_item_01)
#Q14.b.2
set_seed = 14
random.seed(set_seed)
np.random.seed(set_seed)

algorithm = KNNBasic(sim_options = {
'name':'cosine',
'user_based': False
})
name_item_02 = "cosine"
per_item_02 = cross_validate(algorithm, data, measures=['RMSE', 'MAE'], cv=3, verbose=True)
print("Performance:",per_item_02)
#Q14.b.3
set_seed = 15
random.seed(set_seed)
np.random.seed(set_seed)

algorithm = KNNBasic(sim_options = {
'name':'pearson',
'user_based': False
})
name_item_03 = "pearson"
per_item_03 = cross_validate(algorithm, data, measures=['RMSE', 'MAE'], cv=3, verbose=True)
print("Performance:",per_item_03)
Name_User = (name_user_01, name_user_02, name_user_03)
User_rmse = (per_user_01['test_rmse'].mean(), per_user_02['test_rmse'].mean(), per_user_03['test_rmse'].mean())
User_mae = (per_user_01['test_mae'].mean(), per_user_02['test_mae'].mean(), per_user_03['test_mae'].mean())

Name_Item = (name_item_01, name_item_02, name_item_03)
Item_rmse = (per_item_01['test_rmse'].mean(), per_item_02['test_rmse'].mean(), per_item_03['test_rmse'].mean())
Item_mae = (per_item_01['test_mae'].mean(), per_item_02['test_mae'].mean(), per_item_03['test_mae'].mean())

a = np.arange(3)
ax1 = pic.subplot(1,1,1)
b=0.2

print(Name_User)
print("User based RMSE values are ", User_rmse)
print("Item based RMSE values are ", Item_rmse)

pic.bar(a, User_rmse, b)
pic.bar(a + b, Item_rmse,b)

pic.xticks(a + b / 2, Name_User, rotation='vertical')
pic.show()
#Q15.a
measure = []
method =''
k_val=(5,10,15,20,25)
for kvalue in k_val:

    for algorithm in [(KNNBasic(k_val=kvalue, sim_options = {'method':'MSD', 'user_based': True }))]:
    
       
  
        performance = cross_validate(algorithm, data, measures=['RMSE'], cv=3, verbose=False)
   
        data_temp = pd.DataFrame.from_dict(performance).mean(axis=0)
        data_temp = data_temp.append(pd.Series([str(algorithm).split(' ')[0].split('.')[-1]], index=['Algorithm']))
        measure.append(data_temp)
    
UserBased_Dataset=pd.DataFrame(measure).set_index('Algorithm').sort_values('test_rmse')
print(UserBased_Dataset)
pic.bar(k_val,UserBased_Dataset.test_rmse,color='blue',align='center', alpha=0.5)
pic.show()
#Q15.b
measure = []
for kvalue in k_val:

    for algorithm in [(KNNBasic(k_val=kvalue, sim_options = {'name':'MSD', 'user_based': False }))]:
        performance = cross_validate(algorithm, data, measures=['RMSE'], cv=3, verbose=False)
        data_temp = pd.DataFrame.from_dict(performance).mean(axis=0)
        data_temp = data_temp.append(pd.Series([str(algorithm).split(' ')[0].split('.')[-1]], index=['Algorithm']))
        measure.append(data_temp)
    
ItemBased_Dataset=pd.DataFrame(measure).set_index('Algorithm').sort_values('test_rmse')
print(ItemBased_Dataset)
pic.bar(k_val,ItemBased_Dataset.test_rmse,color='green',align='center', alpha=0.5)
pic.show()
