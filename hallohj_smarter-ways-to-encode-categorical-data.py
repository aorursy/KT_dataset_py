import pandas as pd



train = pd.read_csv('../input/train.csv', nrows=None)

train = train.sample(frac=0.2)



#categorical한 데이터들이 정말 integer로 되어있는지 확인

cat_cols = [col for col in train.columns if 'cat' in col]

# train[cat_cols[0]].value_counts()



#categorical data들의 카테고리 수 보기

for col in cat_cols:

    print(col, train[col].nunique())