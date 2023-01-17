import pandas as pd
breeds = pd.read_csv('../input/breed_labels.csv')
colors = pd.read_csv('../input/color_labels.csv')
states = pd.read_csv('../input/state_labels.csv')
train = pd.read_csv('../input/train/train.csv')
test = pd.read_csv('../input/test/test.csv')
new_train = train.copy()
breeds.drop('Type', axis=1, inplace=True)

breeds1 = breeds.copy()
breeds1.columns = ['Breed1', 'BreedName1']
breeds2 = breeds.copy()
breeds2.columns = ['Breed2', 'BreedName2']
new_train = pd.merge(new_train, breeds1, how='left', on='Breed1')
new_train = pd.merge(new_train, breeds2, how='left', on='Breed2')

new_train
breeds.head()
pd.concat()
# breeds['Type'].value_counts()
# #неудачная попытка через .loc
# new_train.loc[new_train.Breed1.astype(int).isin(breeds.BreedID), 'Breed1'] = breeds.BreedName
# new_train.loc[new_train.Breed2.astype(int).isin(breeds.BreedID), 'Breed2'] = breeds.BreedName
# new_train.loc[new_train.Color1.astype(int).isin(colors.ColorID), 'Color1'] = colors.ColorName
# new_train.loc[new_train.Color2.astype(int).isin(colors.ColorID), 'Color2'] = colors.ColorName
# new_train.loc[new_train.Color3.astype(int).isin(colors.ColorID), 'Color3'] = colors.ColorName
# new_train.loc[new_train.State.astype(int).isin(states.StateID), 'State'] = states.StateName

# new_train
# #мердж одной колонки - успешно
# new_train = pd.merge(train, breeds,left_on=['Breed1'], right_on = ['BreedID'], how = 'left')
# new_train.drop(['Breed1'], axis=1, inplace = True)
# new_train.rename(index=str, columns={'BreedName':'Breed1'}, inplace = True)

# new_train
#мердж по двум колонкам одного типа - не успешно. После выполнения второго абзаца, Breed1 восстанавливается до исходного состояния (почему?) Аналогичная проблема с колонками Color
new_train = pd.merge(new_train,breeds,left_on=['Breed1'], right_on = ['BreedID'], how = 'left')
new_train.drop(['Breed1'], axis=1, inplace = True)
new_train.rename(index=str, columns={'BreedName':'Breed1'}, inplace = True)

new_train = pd.merge(new_train,breeds,left_on=['Breed2'], right_on = ['BreedID'], how = 'left')
new_train.drop(['Breed2'], axis=1, inplace = True)
new_train.rename(index=str, columns={'BreedName':'Breed2'}, inplace = True)

new_train












