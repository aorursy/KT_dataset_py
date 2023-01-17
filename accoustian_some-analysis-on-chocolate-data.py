import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline
mainDf = pd.read_csv('../input/flavors_of_cacao.csv')
mainDf.columns=['company','species','REF','review_year','cocoa_p','company_location','rating','bean_typ','country']
mainDf.head(5)
choko= mainDf
d = mainDf

d.cocoa_p = d.cocoa_p.str.replace('%','')

d.cocoa_p = pd.to_numeric(d.cocoa_p)
d.info()
d.cocoa_p.hist(bins=50)
d.boxplot(column='rating')
d.describe()
d.bean_typ.value_counts()
d.country.value_counts()