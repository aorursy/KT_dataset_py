import pandas as pd
import numpy as np
import sqlite3 as sql

conn = sql.connect('../input/database.sqlite')
column_list = (pd.read_sql_query("pragma table_info(Scorecard)", conn))
column_list = list(column_list["name"])
scorecard = pd.read_sql_query("""
SELECT INSTNM, AVGFACSAL, Year, ADM_RATE_ALL, SAT_AVG_ALL,
SATVRMID, SATMTMID, SATWRMID,
mn_earn_wne_p10, mn_earn_wne_p6, mn_earn_wne_p7, mn_earn_wne_p8, mn_earn_wne_p9,
COSTT4_A, COSTT4_P
FROM Scorecard
""", conn)
scorecard.INSTNM = scorecard.INSTNM.str.lower()
#dummy = scorecard.SAT_AVG_ALL.notnull() & scorecard.SATVRMID.notnull() & scorecard.SATMTMID.notnull() & scorecard.SATWRMID.notnull()
#scorecard = scorecard.loc[dummy]
SAT_data = scorecard.loc[:,["INSTNM", "SAT_AVG_ALL", "SATVRMID", "SATMTMID", "SATWRMID", "AVGFACSAL", "Year"]].sort_values(by = "INSTNM")
dummy = SAT_data.SAT_AVG_ALL.notnull() & SAT_data.SATVRMID.notnull() & SAT_data.SATMTMID.notnull() & SAT_data.SATWRMID.notnull()
SAT_data = SAT_data.loc[dummy]
SAT_data_2010 = SAT_data[SAT_data.Year == 2010]
SAT_data_2011 = SAT_data[SAT_data.Year == 2011]
SAT_data_2012 = SAT_data[SAT_data.Year == 2012]
SAT_data_2013 = SAT_data[SAT_data.Year == 2013]
%matplotlib inline
import matplotlib.pyplot as plt
f, axarr = plt.subplots(2, 2)
axarr[0, 0].scatter(SAT_data_2010.SATMTMID, SAT_data_2010.AVGFACSAL)
axarr[0, 0].set_title('SAT Math')
axarr[0, 1].scatter(SAT_data_2010.SATVRMID, SAT_data_2010.AVGFACSAL)
axarr[0, 1].set_title('SAT Verbal')
axarr[1, 0].scatter(SAT_data_2010.SATWRMID, SAT_data_2010.AVGFACSAL)
axarr[1, 0].set_title('SAT Writing')
axarr[1, 1].scatter(SAT_data_2010.SAT_AVG_ALL, SAT_data_2010.AVGFACSAL)
axarr[1, 1].set_title('SAT Average')
plt.setp([a.get_yticklabels() for a in axarr[:, 1]], visible=False)
f, axarr = plt.subplots(2, 2)
axarr[0, 0].scatter(SAT_data_2010.SAT_AVG_ALL, SAT_data_2010.AVGFACSAL)
axarr[0, 0].set_title('2010')
axarr[0, 1].scatter(SAT_data_2011.SAT_AVG_ALL, SAT_data_2011.AVGFACSAL)
axarr[0, 1].set_title('2011')
axarr[1, 0].scatter(SAT_data_2012.SAT_AVG_ALL, SAT_data_2012.AVGFACSAL)
axarr[1, 0].set_title('2012')
axarr[1, 1].scatter(SAT_data_2013.SAT_AVG_ALL, SAT_data_2013.AVGFACSAL)
axarr[1, 1].set_title('2013')
plt.setp([a.get_yticklabels() for a in axarr[:, 1]], visible=False)
plt.setp([a.get_xticklabels() for a in axarr[0, :]], visible=False)
EARN_FACSAL_COST = scorecard.loc[:,['AVGFACSAL', 'mn_earn_wne_p10', 'mn_earn_wne_p6',
       'mn_earn_wne_p7', 'mn_earn_wne_p8', 'mn_earn_wne_p9', 'COSTT4_A',
       'COSTT4_P', "Year"]]
dummy = EARN_FACSAL_COST.AVGFACSAL.notnull() & EARN_FACSAL_COST.mn_earn_wne_p6.notnull()
EARN_p6_FACSAL = EARN_FACSAL_COST.loc[dummy,["AVGFACSAL", "mn_earn_wne_p6", "Year"]]
EARN_p6_FACSAL = EARN_p6_FACSAL[EARN_p6_FACSAL.mn_earn_wne_p6 != "PrivacySuppressed"]
EARN_p6_FACSAL.mn_earn_wne_p6 = EARN_p6_FACSAL.mn_earn_wne_p6.astype(float)
EARN_p6_FACSAL.AVGFACSAL = EARN_p6_FACSAL.AVGFACSAL.astype(float)
EARN_p6_FACSAL_2009 = EARN_p6_FACSAL[EARN_p6_FACSAL.Year == 2009]
EARN_p6_FACSAL_2008 = EARN_p6_FACSAL[EARN_p6_FACSAL.Year == 2008]
EARN_p6_FACSAL_2007 = EARN_p6_FACSAL[EARN_p6_FACSAL.Year == 2007]
EARN_p6_FACSAL_2006 = EARN_p6_FACSAL[EARN_p6_FACSAL.Year == 2006]
EARN_p6_FACSAL_2005 = EARN_p6_FACSAL[EARN_p6_FACSAL.Year == 2005]
plt.scatter(EARN_p6_FACSAL_2005.mn_earn_wne_p6, EARN_p6_FACSAL_2005.AVGFACSAL);
plt.scatter(EARN_p6_FACSAL_2006.mn_earn_wne_p6, EARN_p6_FACSAL_2006.AVGFACSAL);
plt.scatter(EARN_p6_FACSAL_2007.mn_earn_wne_p6, EARN_p6_FACSAL_2007.AVGFACSAL);
plt.scatter(EARN_p6_FACSAL_2008.mn_earn_wne_p6, EARN_p6_FACSAL_2008.AVGFACSAL);
plt.scatter(EARN_p6_FACSAL_2009.mn_earn_wne_p6, EARN_p6_FACSAL_2009.AVGFACSAL)
