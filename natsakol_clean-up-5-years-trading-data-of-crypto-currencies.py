import pandas as pd 
import warnings
warnings.filterwarnings("ignore")

#read csv file and save as 'raw'
raw = pd.read_csv('../input/CryptocoinsHistoricalPrices.csv')
raw.info()
#describe raw dataset sorted by date descendly
raw.sort_values(by='Date',ascending=False)
clean=raw.dropna(axis=0, inplace=False, thresh=7) #delete the row which containing occupied columns less than 7 columns 
clean.sort_values(by='Date',ascending=False) #describe dataset after deleting the unreadable rows
#delete 'Unnamed: 0' columns
#delete non-numeric open price
#delete non-numeric close price
clean=clean.drop(['Unnamed: 0','Open.','Close..'], axis=1, inplace=False)
#show the first row of clean2 dataframe
clean.head(1)
clean= clean[clean['Market.Cap']!='-']
clean['Market.Cap'] = clean['Market.Cap'].str.replace(',', '')
clean['Volume'] = clean['Volume'].str.replace(',', '')
#list of new columns' order
cols= [ 'coin','Date','Open','Close','High','Low','Volume','Market.Cap','Delta']
#rearrange order
clean = clean[cols]
#show the first row of dataset after reordering columns
clean.head(1)
#list object columns
obj = clean.select_dtypes(include = object).columns.tolist()
#exclude coin and date columns from the list
obj.remove('coin')
obj.remove('Date')
#Convert object columns to numeric columns
clean[obj] = clean[obj].convert_objects(convert_numeric = True)
clean.info()
clean['Market.Cap']=clean['Market.Cap'].astype(float)
clean.info()