# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

# import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import pandas as pd

housing_df = pd.read_csv("../input/generaldatasets/WestRoxbury.csv")
housing_df.shape #	find	the	dimension	of	data	frame
housing_df.head()   	 #	show	the	first	five	rows
print(housing_df)	 #	show	all	the	data
#	Rename	columns:	replace	spaces	with	’_’	to	allow	dot	notation	

housing_df = housing_df.rename(columns={'TOTAL VALUE':'TOTAL_VALUE'})
housing_df.columns=[s.strip().replace(' ','_') for s in housing_df.columns] 

#	Practice	showing	the	first	four	rows	of	the	data

housing_df.loc[0:3]	 #	loc[a:b]	gives	rows	a	to	b,	inclusive	
housing_df.iloc[0:4]	 #	iloc[a:b]	gives	rows	a	to	b-1

#	Different	ways	of	showing	the	first	10	values	in	column	 TOTAL_VALUE

housing_df['TOTAL_VALUE'].iloc[0:10]

housing_df.iloc[0:10]['TOTAL_VALUE']

housing_df.iloc[0:10].TOTAL_VALUE	 #	use	dot	notation	if	the	
#	Show	the	fifth	row	of	the	first	10	columns

housing_df.iloc[4][0:10]
housing_df.iloc[4,	0:10]
housing_df.iloc[4:5,	0:10]	 #	use	a	slice	to	return	a	data	frame	
#	The	axis	argument	specifies	the	dimension	along	which	the

#	concatenation	happens,	0=rows,	1=columns.

pd.concat([housing_df.iloc[4:6,0:2],	housing_df.iloc[4:6,4:6]],	

axis=1)

#	To	specify	a	full	column,	use:

housing_df.iloc[:,0:1]

housing_df.TOTAL_VALUE

housing_df['TOTAL_VALUE'][0:10]	 #	show	the	first	10	rows	of	the	first	column
#	Descriptive	statistics

print('Number	of	rows	',	len(housing_df['TOTAL_VALUE']))	#	show length	of	first	column

print('Mean of TOTAL_VALUE', housing_df['TOTAL_VALUE'].mean())	#	 show	mean	of	column
housing_df.describe()	#	show	summary	statistics	for	each	column
import	numpy	as	np

import	pandas	as	pd

from	sklearn.model_selection	import	train_test_split

from	sklearn.metrics	import	r2_score

from	sklearn.linear_model	import	LinearRegression
#	random	sample	of	5	observations

housing_df.sample(5)

#	oversample	houses	with	over	10	rooms

weights	=	[0.9	if	rooms	>	10	else	0.01	for	rooms	in	housing_df.ROOMS]

housing_df.sample(5,	weights=weights)
housing_df.columns		#	print	a	list	of	variables
#	REMODEL	needs	to	be	converted	to	a	categorical	variable

housing_df.REMODEL	=	housing_df.REMODEL.astype('category')
housing_df.REMODEL.cat.categories		#	Show	number	of	categories
housing_df.REMODEL.dtype		#	Check	type	of	converted	variab
#	To	illustrate	missing	data	procedures,	we	first	convert	a	few entries	for

#	bedrooms	to	NA’s.	Then	we	impute	these	missing	values	using	the	median	of	the

#	remaining	values.

missingRows	=	housing_df.sample(10).index

housing_df.loc[missingRows,	'BEDROOMS']	=	np.nan

print('Number	of	rows	with	valid	BEDROOMS	values	after	setting	to	NAN:	',	housing_df['BEDROOMS'].count())

#	remove	rows	with	missing	values

reduced_df	=	housing_df.dropna()

print('Number	of	rows	after	removing	rows	with	missing	values:	',	

len(reduced_df))

#	replace	the	missing	values	using	the	median	of	the	remaining	 values.

medianBedrooms	=	housing_df['BEDROOMS'].median()

housing_df.BEDROOMS	= housing_df.BEDROOMS.fillna(value=medianBedrooms)

print('Number	of	rows	with	valid	BEDROOMS	values	after	filling	NA	values:	',	housing_df['BEDROOMS'].count())
from	sklearn.preprocessing	import	MinMaxScaler,	StandardScaler

df	=	housing_df.copy()

#	Normalizing	a	data	frame

#	pandas:

norm_df	=	(housing_df	-	housing_df.mean())	/	housing_df.std()

#	scikit-learn:

scaler	=	StandardScaler()

norm_df	=	pd.DataFrame(scaler.fit_transform(housing_df),index=housing_df.index,	columns=housing_df.columns)

#	the	result	of	the	transformation	is	a	numpy	array,	we	convert	it	into	a	dataframe

#	Rescaling	a	data	frame

#	pandas:

norm_df	=	(housing_df	-	housing_df.min())	/	(housing_df.max()	-housing_df.min())

#	scikit-learn:

scaler	=	MinMaxScaler()

norm_df	=	pd.DataFrame(scaler.fit_transform(housing_df),	index=housing_df.index,	columns=housing_df.columns)