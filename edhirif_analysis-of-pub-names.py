# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

df = pd.read_csv('../input/open_pubs.csv')

from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
scotland = ['AB1', 'AB2', 'AB23', 'AB3', 'AB30','AB31', 'AB32', 'AB33', 'AB34', 'AB35', 'AB36', 'AB41', 'AB42', 'AB43', 'AB44', 'AB45', 'AB51', 'AB52', 'AB53', 'AB54', 'AB55', 'DD10', 'DD9',

'DD10', 'DD11', 'DD2', 'DD3', 'DD4', 'DD5', 'DD7', 'DD8', 'DD9', 'PH11', 'PH12',

'FK17', 'G82', 'G83', 'G84', 'PA20', 'PA21', 'PA22', 'PA23', 'PA24', 'PA25', 'PA26', 'PA27', 'PA28', 'PA29', 'PA30', 'PA31', 'PA32', 'PA33', 'PA34', 'PA35', 'PA36', 'PA37', 'PA38', 'PA41',

'PA42', 'PA43', 'PA44', 'PA45', 'PA46', 'PA47', 'PA48', 'PA49', 'PA60', 'PA61', 'PA62', 'PA63', 'PA64', 'PA65', 'PA66', 'PA67', 'PA68', 'PA70', 'PA71', 'PA72', 'PA73', 'PA74', 'PA75', 'PA76',

'PA77', 'PA78',

'DG7', 'KA1', 'KA10', 'KA11', 'KA12', 'KA13', 'KA14', 'KA15', 'KA16', 'KA17', 'KA18', 'KA19', 'KA2', 'KA20', 'KA21', 'KA22', 'KA23', 'KA24', 'KA25', 'KA26', 'KA27', 'KA28', 'KA29', 'KA3',

'KA30', 'KA4', 'KA5', 'KA6', 'KA7', 'KA8', 'KA9', 'PA17',

'FK10', 'FK11', 'FK12', 'FK13', 'FK14',

'DG1', 'DG10', 'DG11', 'DG12', 'DG13', 'DG14', 'DG16', 'DG2', 'DG3', 'DG4', 'DG5', 'DG6', 'DG7', 'DG8', 'DG9', 'KA6', 'ML12',

'G60', 'G61', 'G62', 'G64', 'G65', 'G66', 'G81', 'G82', 'G83',

'DD1', 'DD2', 'DD3', 'DD5',

'EH21', 'EH22', 'EH31', 'EH32', 'EH33', 'EH34', 'EH35', 'EH36', 'EH39', 'EH40', 'EH41', 'EH42', 'TD13',

'EH1', 'EH10', 'EH12', 'EH13', 'EH14', 'EH15', 'EH16', 'EH17', 'EH2', 'EH20', 'EH21', 'EH28', 'EH29', 'EH3', 'EH30', 'EH4', 'EH5', 'EH6', 'EH7', 'EH8', 'EH9',

'FK1', 'FK10', 'FK11', 'FK12', 'FK13', 'FK14', 'FK15', 'FK16', 'FK17', 'FK18', 'FK19', 'FK2', 'FK20', 'FK21', 'FK3', 'FK4', 'FK5', 'FK6', 'FK7', 'FK8', 'FK9',

'DD6', 'FK10', 'KY1', 'KY10', 'KY11', 'KY12', 'KY13', 'KY14', 'KY15', 'KY16', 'KY2', 'KY3', 'KY4', 'KY5', 'KY6', 'KY7', 'KY8', 'KY9',

'G1', 'G11', 'G12', 'G14', 'G15', 'G2', 'G20', 'G21', 'G32', 'G40', 'G41', 'G42', 'G43', 'G45', 'G51', 'G53', 'G76',

'AB37', 'IV1', 'IV10', 'IV11', 'IV12', 'IV13', 'IV14', 'IV15', 'IV16', 'IV17', 'IV18', 'IV19', 'IV2', 'IV20', 'IV21', 'IV22', 'IV23', 'IV24', 'IV25', 'IV26', 'IV27', 'IV28', 'IV3', 'IV4', 'IV40', 'IV41', 'IV42',

'IV43', 'IV44', 'IV45', 'IV46', 'IV47', 'IV48', 'IV49', 'IV5', 'IV51', 'IV52', 'IV53', 'IV54', 'IV55', 'IV56', 'IV6', 'IV7', 'IV8', 'IV9', 'KW1', 'KW10', 'KW11', 'KW12', 'KW13','KW14', 'KW2',

'KW3', 'KW5', 'KW6',

'KW7', 'KW8', 'KW9', 'PA34', 'PA38', 'PA39', 'PA40', 'PH19', 'PH20', 'PH21', 'PH22', 'PH23', 'PH24', 'PH25', 'PH26', 'PH30', 'PH31', 'PH32', 'PH33', 'PH34', 'PH35', 'PH36', 'PH37',

'PH38', 'PH39', 'PH40', 'PH41', 'PH42', 'PH43', 'PH44',

'PA10', 'PA11', 'PA13', 'PA14', 'PA15', 'PA16', 'PA18', 'PA19',

'EH46', 'EH55', 'G33', 'G65', 'G66', 'G67', 'G68', 'G69', 'G71', 'G72', 'G73', 'G74', 'G75', 'ML1', 'ML10', 'ML11', 'ML12', 'ML2', 'ML3', 'ML4', 'ML5', 'ML6', 'ML7', 'ML8', 'ML9',

'EH10', 'EH18', 'EH19', 'EH20', 'EH22', 'EH23', 'EH24', 'EH25', 'EH26', 'EH37', 'EH46',

'AB37', 'AB38', 'AB56', 'IV30', 'IV31', 'IV32', 'IV36', 'PH26',

'KW1', 'KW15', 'KW16', 'KW17',

'DD2', 'FK14', 'FK15', 'FK19', 'FK21', 'KY13', 'PH1', 'PH10', 'PH11', 'PH12', 'PH13', 'PH14', 'PH15', 'PH16', 'PH17', 'PH18', 'PH2', 'PH3', 'PH4', 'PH5', 'PH6', 'PH7', 'PH8', 'PH9',

'G46', 'G77', 'G78', 'PA1', 'PA10', 'PA11', 'PA12', 'PA14', 'PA3', 'PA4', 'PA5', 'PA6', 'PA7', 'PA8', 'PA9',

'EH26', 'EH38', 'EH43', 'EH44','EH45', 'EH46', 'ML12', 'TD1', 'TD10', 'TD11', 'TD12', 'TD13', 'TD14', 'TD15', 'TD2', 'TD3', 'TD4', 'TD5', 'TD6', 'TD7', 'TD8', 'TD9',

'ZE1', 'ZE2', 'ZE3',

'FK15', 'FK16', 'FK17', 'FK18', 'FK19', 'FK20', 'FK21', 'FK6', 'FK7', 'FK8', 'FK9', 'G63',

'EH27', 'EH30', 'EH47', 'EH48', 'EH49', 'EH52', 'EH53', 'EH54', 'EH55',

'HS1', 'HS2', 'HS3', 'HS4', 'HS5', 'HS6', 'HS7', 'HS8', 'HS9']
df['area'],df['sub_area'] = df['postcode'].str.split(' ').str

df_scotland = df[df['area'].isin(scotland)]


df['name'].value_counts().head(10)
df_scotland['name'].value_counts().head(10)