# data transformation
import pandas as pd

# viz
import seaborn as sns
import matplotlib.pyplot as plt

# used to effciently read all stock data
from joblib import Parallel, delayed
from glob import glob
from os.path import basename


def read_single(f):
    """read a single file, return a tuple with the stock name and close prices"""
    raw = pd.read_csv(f)
    
    raw['Date'] = pd.to_datetime(raw['Date'])
    raw.set_index('Date', inplace=True)
    
    stock_name = basename(f).split('.')[0]
    return stock_name, raw['Close']


# parallel load all the close prices from "1 Day/Stocks"
res = Parallel(n_jobs=4)(delayed(read_single)(f)
                         for f in glob('../input/Data/1 Day/Stocks/*.us.txt'))

df = pd.DataFrame()
for stock_name, close_price in res:
    df[stock_name] = close_price
res = None

df.head()
# Define the indicators that we care about
def bbands(df, periods=30, numsd=2):
    """ returns average, upper band, and lower band"""
    r = df.rolling(periods)
    ave = r.mean()
    sd = r.std()
    upband = ave + (sd*numsd)
    dnband = ave - (sd*numsd)
    return ave, upband, dnband

def relative_strength(df, n=14):
    delta = df.diff()
    dUp, dDown = delta.copy(), delta.copy()
    dUp[dUp < 0] = 0
    dDown[dDown > 0] = 0

    RolUp = dUp.rolling(n).mean()
    RolDown = dDown.rolling(n).mean().abs()

    RS = RolUp / RolDown

    return 100. - 100./(1. + RS)

rsi = relative_strength(df)
sma, up, dn = bbands(df)
pct_b = (df - dn) / (up - dn)
pct_b.tail()
for indicator in 'atvi,adbe,akam,alxn,goog,googl,amzn,aal,amgn,adi,aapl,amat,adsk,adp,bidu,biib,bmrn,avgo,ca,celg,cern,chtr,chkp,ctas,csco,ctxs,ctsh,cmcsa,cost,csx,ctrp,xray,disca,disck,dish,dltr,ebay,ea,expe,esrx,fb,fast,fisv,gild,has,hsic,holx,ilmn,incy,intc,intu,isrg,jd,klac,lrcx,lbtya,lbtyk,lila,lilak,lvnta,qvca,mar,mat,mxim,mchp,mu,msft,mdlz,mnst,myl,ntes,nflx,nclh,nvda,nxpi,orly,pcar,payx,pypl,qcom,regn,rost,sbac,stx,shpg,siri,swks,sbux,symc,tmus,tsla,txn,khc,pcln,tsco,trip,fox,foxa,ulta,vrsk,vrtx,viab,vod,wba,wdc,xlnx'.split(','):
    g = sns.jointplot(x=rsi[indicator], y=pct_b[indicator], kind="kde", shade=True)
    plt.title(indicator)
    g.set_axis_labels("rsi", "pct_b")
