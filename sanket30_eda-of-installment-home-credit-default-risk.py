import pandas as pd
# import import_ipynb
# import Functions as f
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns
plt.style.use('bmh')
plt.style.use('ggplot')
plt.rcParams['figure.figsize'] = 20, 10
installments_paymentss=pd.read_csv("../input/installments_payments.csv")

installments=installments_paymentss.copy()                              
installments['Days Extra Taken']=installments['DAYS_INSTALMENT']-installments['DAYS_ENTRY_PAYMENT']
installments['AMT_INSTALMENT_difference']=installments['AMT_INSTALMENT']-installments['AMT_PAYMENT']
installments.columns.tolist()
installmentcol=['SK_ID_CURR',
 'NUM_INSTALMENT_VERSION',
 'NUM_INSTALMENT_NUMBER',
 'Days Extra Taken',
 'AMT_INSTALMENT_difference']
installments=installments[installmentcol]
installments.shape
tempinstall=(installments.groupby(['SK_ID_CURR']).mean()).round()
tempinstall['SK_ID_CURR']=tempinstall.index.values
installment_df=tempinstall.reset_index(drop=True)
installment_df.shape
installment_df.describe().round()
installment_df.isna().sum()
installment_df['Days Extra Taken']=installment_df['Days Extra Taken'].fillna(installment_df['Days Extra Taken'].mean())
installment_df['AMT_INSTALMENT_difference']=installment_df['AMT_INSTALMENT_difference'].fillna(installment_df['AMT_INSTALMENT_difference'].mean())
installment_df[installment_df['AMT_INSTALMENT_difference']==146146]
correlation = installment_df.corr()
plt.figure(figsize=(15,15))
sns.heatmap(correlation, vmax=1, square=True,annot=True,cmap='cubehelix')
plt.title('Correlation between different fearures')
from sklearn.preprocessing import Normalizer
normalized_installment = pd.DataFrame(Normalizer().fit_transform(installment_df))
normalized_installment.columns=['SK_ID_CURR',
 'NUM_INSTALMENT_VERSION',
 'NUM_INSTALMENT_NUMBER',
 'Days Extra Taken',
 'AMT_INSTALMENT_difference']
correlation = normalized_installment.corr()
plt.figure(figsize=(15,15))
sns.heatmap(correlation, vmax=1, square=True,annot=True,cmap='cubehelix')
plt.title('Correlation between different fearures')
