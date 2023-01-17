import numpy as np
import pandas as pd


df = pd.read_csv('../input/BreadBasket_DMS.csv')
df.head(10)
df['year'] = pd.DatetimeIndex(df['Date']).year
df['month'] = pd.DatetimeIndex(df['Date']).month
df['day'] = pd.DatetimeIndex(df['Date']).day
df['hour'] = pd.DatetimeIndex(df['Time']).hour
df['minute'] = pd.DatetimeIndex(df['Time']).minute
df.head()
df_Normalized = df.groupby(['Transaction', 'year', 'month', 'day', 'hour', 'minute'])['Item'].count().reset_index()
df_Normalized.head(20)
df_Normalized.count()
df_Normalized.tail()
import seaborn as sns
sns.countplot(data=df_Normalized, x="month")
sns.countplot(data=df, x="month")
sns.countplot(data=df_Normalized, x="hour")
sns.countplot(data=df, x="hour")
df_ItemHour = df.groupby(['hour', 'Item'])['Transaction'].count().reset_index()
df_ItemHour.head(10)
df_ItemHour_Pivot = df_ItemHour.pivot(index='Item', columns='hour', values='Transaction')
df_ItemHour_Pivot = df_ItemHour_Pivot.fillna(0)
df_ItemHour_Pivot.head(10)
import matplotlib.pyplot as plt
fig,ax=plt.subplots(figsize=(10,80))
sns.heatmap(df_ItemHour_Pivot, vmin=5, vmax=200, yticklabels=True, ax=ax, cmap="YlGnBu")
df_ItemHour_Focused = df_ItemHour_Pivot.drop(['Adjustment', 'Afternoon with the baker', 'Argentina Night', 'Art Tray', 'Bacon', 'Bakewell', 'Bare Popcorn', 'Basket', 'Bowl Nic Pitt', 'Bread Pudding', 'Brioche and salami', 'Caramel bites', 'Cherry me Dried fruit', 'Chicken sand', 'Chimichurri Oil', 'Chocolates', 'Christmas common','Crepes', 'Crisps', 'Duck egg', 'Dulce de Leche', 'Eggs', 'Empanadas', 'Fairy Doors', 'Gift voucher', 'Gingerbread syrup', 'Granola', 'Hack the stack', 'Honey', 'Kids biscuit', 'Lemon and coconut', 'Mighty Protein', 'Mortimer', 'Muesli', 'My-5 Fruit Shoot', 'Nomad bag', 'Olum & polenta', 'Panatone', 'Pick and Mix Bowls', 'Pintxos', 'Polenta', 'Postcard', 'Raspberry shortbread sandwich', 'Raw bars', 'Siblings', 'Spread', 'Tacos/Fajita', 'Tartine', 'The BART', 'Tshirt', 'Valentine\'s card', 'Vegan Feast', 'Victorian Sponge'])
df_ItemHour_Focused
fig,ax=plt.subplots(figsize=(10,20))
sns.heatmap(df_ItemHour_Focused, vmin=3, vmax=200, yticklabels=True, ax=ax, cmap="YlGnBu")
df_ItemDay = df.groupby(['day', 'Item'])['Transaction'].count().reset_index()
df_ItemDay.head(10)
df_ItemDay_Pivot = df_ItemDay.pivot(index='Item', columns='day', values='Transaction')
df_ItemDay_Pivot = df_ItemDay_Pivot.fillna(0)
df_ItemDay_Focused = df_ItemDay_Pivot.drop(['Adjustment', 'Afternoon with the baker', 'Argentina Night', 'Art Tray', 'Bacon', 'Bakewell', 'Bare Popcorn', 'Basket', 'Bowl Nic Pitt', 'Bread Pudding', 'Brioche and salami', 'Caramel bites', 'Cherry me Dried fruit', 'Chicken sand', 'Chimichurri Oil', 'Chocolates', 'Christmas common','Crepes', 'Crisps', 'Duck egg', 'Dulce de Leche', 'Eggs', 'Empanadas', 'Fairy Doors', 'Gift voucher', 'Gingerbread syrup', 'Granola', 'Hack the stack', 'Honey', 'Kids biscuit', 'Lemon and coconut', 'Mighty Protein', 'Mortimer', 'Muesli', 'My-5 Fruit Shoot', 'Nomad bag', 'Olum & polenta', 'Panatone', 'Pick and Mix Bowls', 'Pintxos', 'Polenta', 'Postcard', 'Raspberry shortbread sandwich', 'Raw bars', 'Siblings', 'Spread', 'Tacos/Fajita', 'Tartine', 'The BART', 'Tshirt', 'Valentine\'s card', 'Vegan Feast', 'Victorian Sponge'])
df_ItemDay_Focused
fig,ax=plt.subplots(figsize=(15,15))
sns.heatmap(df_ItemDay_Focused, vmin=5, vmax=150, yticklabels=True, ax=ax, cmap="YlGnBu")
df['my_dates'] = pd.to_datetime(df['Date'])
df['day_of_week'] = df['my_dates'].dt.weekday_name
df.head(10)
sns.countplot(data=df, x="day_of_week")
df_ItemDayOfWeek = df.groupby(['day_of_week', 'Item'])['Transaction'].count().reset_index()
df_ItemDayOfWeek.head(10)
df_ItemDayOfWeek_Pivot = df_ItemDayOfWeek.pivot(index='Item', columns='day_of_week', values='Transaction')
df_ItemDayOfWeek_Pivot = df_ItemDayOfWeek_Pivot.fillna(0)
df_ItemDayOfWeek_Focused = df_ItemDayOfWeek_Pivot.drop(['Adjustment', 'Afternoon with the baker', 'Argentina Night', 'Art Tray', 'Bacon', 'Bakewell', 'Bare Popcorn', 'Basket', 'Bowl Nic Pitt', 'Bread Pudding', 'Brioche and salami', 'Caramel bites', 'Cherry me Dried fruit', 'Chicken sand', 'Chimichurri Oil', 'Chocolates', 'Christmas common','Crepes', 'Crisps', 'Duck egg', 'Dulce de Leche', 'Eggs', 'Empanadas', 'Fairy Doors', 'Gift voucher', 'Gingerbread syrup', 'Granola', 'Hack the stack', 'Honey', 'Kids biscuit', 'Lemon and coconut', 'Mighty Protein', 'Mortimer', 'Muesli', 'My-5 Fruit Shoot', 'Nomad bag', 'Olum & polenta', 'Panatone', 'Pick and Mix Bowls', 'Pintxos', 'Polenta', 'Postcard', 'Raspberry shortbread sandwich', 'Raw bars', 'Siblings', 'Spread', 'Tacos/Fajita', 'Tartine', 'The BART', 'Tshirt', 'Valentine\'s card', 'Vegan Feast', 'Victorian Sponge'])
column_order = ['Sunday', 'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday']
df_ItemDayOfWeek_Focused = df_ItemDayOfWeek_Focused.reindex(column_order, axis=1)
df_ItemDayOfWeek_Focused
fig,ax=plt.subplots(figsize=(15,15))
sns.heatmap(df_ItemDayOfWeek_Focused, vmin=5, vmax=250, yticklabels=True, ax=ax, cmap="YlGnBu")
