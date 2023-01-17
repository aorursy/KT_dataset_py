import pandas as pd
root = '/kaggle/input/scl-dummy/'
data = pd.read_csv(root+'Dummy data.csv')
data
x = [[i,i+2] for i in data['id']]
data = pd.DataFrame(x, columns=['id','new_number'])
data
data.to_csv('submission.csv',header=True,index=False)