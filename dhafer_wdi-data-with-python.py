!pip install wbdata
import wbdata
wbdata.get_source()
wbdata.get_indicator(source=1)
wbdata.search_countries("tun")
import time

from datetime import date
date.today()
data_date = (date(2010, 1, 1), date.today())

data_date
indicators = {'NY.GNP.PCAP.CD':'GNI per Capita'}
wbdata.get_data(indicator='NY.GNP.PCAP.CD' , country='TUN', data_date=data_date)
dd=wbdata.get_data(indicator='NY.GNP.PCAP.CD' , country='TUN' , data_date=data_date)
import pandas as pd
dd=pd.DataFrame(dd)
dd.head()
dd.shape