import pandas as pd

df = pd.read_csv('../input/train.csv')
print('A.ID B.ID A.eeb9cd3aa B.target')
for a in df[['bd8f989f1','29ab304b9','8dc7f1eb9','ID','eeb9cd3aa']].values:
	if a[0]!=0 and a[1]!=0 and a[2]!=0:
		for b in df[['a75d400b8','1d9078f84','7d287013b','ID','target']].values:
			if b[0]==a[0] and b[1]==a[1] and b[2]==a[2] and a[3]!=b[3]:
				print(a[3],b[3],a[4],b[4])