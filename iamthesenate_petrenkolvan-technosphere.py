df=pd.read_csv('../input/train.csv')

df=df.dropna()

df.shape
tdf=pd.read_csv('../input/test.csv')