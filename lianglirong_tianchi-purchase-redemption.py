import numpy as np
import pandas as pd 
import os
print(os.listdir("../input/purchase redemption data/Purchase Redemption Data"))
#用户信息表
pd_user_profile_table = pd.read_csv("../input/purchase redemption data/Purchase Redemption Data/user_profile_table.csv")
pd_user_profile_table.head()
groupby_city = pd_user_profile_table.groupby("city")
groupby_city.size()
#银行间同业拆放利率（Shibor）表 
pd_mfd_bank_shibor = pd.read_csv("../input/purchase redemption data/Purchase Redemption Data/mfd_bank_shibor.csv")
pd_mfd_bank_shibor.head()

#用户申购赎回数据表
pd_user_balance_table = pd.read_csv("../input/purchase redemption data/Purchase Redemption Data/user_balance_table.csv")
pd_user_balance_table.head()
#余额宝收益表
pd_mfd_day_share_interest = pd.read_csv("../input/purchase redemption data/Purchase Redemption Data/mfd_day_share_interest.csv")
pd_mfd_day_share_interest.head()
#预测表模板
pd_comp_predict_table = pd.read_csv("../input/purchase redemption data/Purchase Redemption Data/comp_predict_table.csv")
pd_comp_predict_table.head()
