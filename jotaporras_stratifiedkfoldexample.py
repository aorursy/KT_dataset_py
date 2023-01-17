from sklearn import model_selection
import pandas as pd
import numpy as np
# Parameters
num_observations = 10000
num_classes = 5
num_folds = 5
# Create a random dataset, let's assume this is our training data

df = pd.DataFrame({
    'input_images': pd.Series(np.random.randint(0,1000000,size=num_observations)),
    'target': np.random.randint(0,num_classes,size=num_observations)
})
df.input_images = df.input_images.astype('str') + ".jpg"
df
# Shuffle the dataset 
df = df.sample(frac=1).reset_index(drop=True)

# Get target values
y = df.target.values
stratified_kfold = model_selection.StratifiedKFold(n_splits=num_folds)

for fold_id, (_, rows) in enumerate(stratified_kfold.split(X=df, y=y)):
    df.loc[rows, 'fold'] = fold_id
df.fold = df.fold.astype(int)

df.head()
def train(df, fold):
    train_fold
def train(df, fold):
    train_fold = df[df.fold==fold]
    test_fold = df[df.fold!=fold]
    
    # ... [use the fold subets to create the train and validation loaders]
    
    # ... [Insert training loop for each epoch.]
    
    # Finally do something like
    torch.save(model.state_dict(),'model_{fold}.pth') # (this won't run)
    
train_fold_1 = df[df.fold==1]
test_fold_1 = df[df.fold!=1]
print(train_fold_1.groupby('target').size())
print(test_fold_1.groupby('target').size())
train_fold_2 = df[df.fold==2]
test_fold_2 = df[df.fold!=2]
print(train_fold_2.groupby('target').size())
print(test_fold_2.groupby('target').size())