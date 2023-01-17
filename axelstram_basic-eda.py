# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))
import matplotlib.pyplot as plt
import json
# Any results you write to the current directory are saved as output.
df = pd.read_csv('../input/train.csv', parse_dates=['date'])
# Create new dataframes from JSON fields

device_df = df.device.apply(json.loads).values.tolist()
device_df = pd.DataFrame.from_records(device_df)

geoNetwork_df = df.geoNetwork.apply(json.loads).values.tolist()
geoNetwork_df = pd.DataFrame.from_records(geoNetwork_df)

totals_df = df.totals.apply(json.loads).values.tolist()
totals_df = pd.DataFrame.from_records(totals_df)

trafficSource_df = df.trafficSource.apply(json.loads).values.tolist()
trafficSource_df = pd.DataFrame.from_records(trafficSource_df)
# Merge with the original DataFrame
df.drop(['device', 'geoNetwork', 'totals', 'trafficSource'], axis=1, inplace=True)
df = pd.concat([df, geoNetwork_df, device_df, totals_df, trafficSource_df], axis=1)
# DataFrame with new fields
df.info()
# There are several of this columns with only one value, so they can be discarded.
df.channelGrouping.value_counts().plot(kind='bar', figsize=(15, 5), rot=70, title="Channel Grouping");
df.date.dt.year.value_counts().plot(kind='bar', figsize=(15, 5), rot=70, title="Year");
df.date.dt.month.value_counts().plot(kind='bar', figsize=(15, 5), rot=70, title="Month");
# Can be discarded
df.socialEngagementType.value_counts().plot(kind='bar', figsize=(15, 5), rot=70, title="socialEngagementType");
df.visitNumber.value_counts()[:20].plot(kind='bar', figsize=(15, 5), rot=70, title="visitNumber");
df.city.value_counts()[:20].plot(kind='bar', figsize=(15, 5), rot=70, title="city");
# Can be discarded
df.cityId.value_counts().plot(kind='bar', figsize=(15, 5), rot=70, title="cityId");
df.continent.value_counts().plot(kind='bar', figsize=(15, 5), rot=70, title="continent");
df.country.value_counts()[:20].plot(kind='bar', figsize=(15, 5), rot=70, title="country");
# Can be discarded
df.latitude.value_counts().plot(kind='bar', figsize=(15, 5), rot=70, title="latitude");
# Can be discarded
df.longitude.value_counts().plot(kind='bar', figsize=(15, 5), rot=70, title="longitude");
# Maybe can merge "not available in demo dataset" with "(not set)?"
df.metro.value_counts()[:20].plot(kind='bar', figsize=(15, 5), rot=70, title="metro");
df.networkDomain.value_counts()[:20].plot(kind='bar', figsize=(15, 5), rot=70, title="networkDomain");
# Can be discarded
df.networkLocation.value_counts().plot(kind='bar', figsize=(15, 5), rot=70, title="networkLocation");
df.region.value_counts()[:20].plot(kind='bar', figsize=(15, 5), rot=70, title="region");
df.subContinent.value_counts()[:20].plot(kind='bar', figsize=(15, 5), rot=70, title="subContinent");
df.browser.value_counts()[:20].plot(kind='bar', figsize=(15, 5), rot=70, title="browser");
# Can be discarded
df.browserSize.value_counts().plot(kind='bar', figsize=(15, 5), rot=70, title="browserSize");
# Can be discarded
df.browserVersion.value_counts().plot(kind='bar', figsize=(15, 5), rot=70, title="browserVersion");
df.deviceCategory.value_counts().plot(kind='bar', figsize=(15, 5), rot=70, title="deviceCategory");
# Can be discarded
df.flashVersion.value_counts().plot(kind='bar', figsize=(15, 5), rot=70, title="flashVersion");
# Warn: Falses are set as NaN's
df['isMobile'] = df['isMobile'].fillna(False)
df.isMobile.value_counts().plot(kind='bar', figsize=(15, 5), rot=70, title="isMobile");
# All this can be discarded
print(df.language.value_counts())
print(df.mobileDeviceBranding.value_counts())
print(df.mobileDeviceInfo.value_counts())
print(df.mobileDeviceMarketingName.value_counts())
print(df.mobileDeviceModel.value_counts())
print(df.mobileInputSelector.value_counts())
print(df.operatingSystemVersion.value_counts())
print(df.screenColors.value_counts())
print(df.screenResolution.value_counts())
print(df.bounces.value_counts())
print(df.newVisits.value_counts())
print(df.visits.value_counts())
print(df.campaignCode.value_counts())

df.operatingSystem.value_counts().plot(kind='bar', figsize=(15, 5), rot=70, title="operatingSystem");
df.hits.value_counts()[:20].plot(kind='bar', figsize=(15, 5), rot=70, title="hits");
df.pageviews.value_counts()[:20].plot(kind='bar', figsize=(15, 5), rot=70, title="pageviews");
df.adContent.value_counts()[:20].plot(kind='bar', figsize=(15, 5), rot=70, title="adContent");
df.campaign.value_counts().plot(kind='bar', figsize=(15, 5), rot=70, title="campaign");
df.keyword.value_counts()[:20].plot(kind='bar', figsize=(15, 5), rot=70, title="keyword");
df.medium.value_counts().plot(kind='bar', figsize=(15, 5), rot=70, title="medium");
df.referralPath.value_counts()[:20].plot(kind='bar', figsize=(15, 5), rot=70, title="referralPath");
df.source.value_counts()[:20].plot(kind='bar', figsize=(15, 5), rot=70, title="source");
adwordsClickInfo = df['adwordsClickInfo'].apply(pd.Series)
adwordsClickInfo.head()
# Maybe some of this can be useful, not sure yet.
print(adwordsClickInfo['criteriaParameters'].value_counts())
print(adwordsClickInfo['page'].value_counts())
print(adwordsClickInfo['slot'].value_counts())
print(adwordsClickInfo['gclId'].value_counts())
print(adwordsClickInfo['adNetworkType'].value_counts())
print(adwordsClickInfo['isVideoAd'].value_counts())

# Mostly Nan's, can be discarded
adwordsClickInfo['targetingCriteria'].isna().sum() / len(adwordsClickInfo)
# Target variable
# Warn: Contains mostly NaN's
df['transactionRevenue'].isna().sum() / len(df)
# I interpret NaN as if there was no transaction.
df['transactionRevenue'] = df.transactionRevenue.fillna(0)

df['transactionRevenue'].value_counts()[:20].plot(kind='bar', figsize=(15, 5), rot=70, title="transactionRevenue");