#İlk olarak çalışmamızda kullanacağımız modülleri import(yükleme-dahil etme) ediyoruz.
import pandas as pd
import numpy as np

#Box-Cox Dönüştürme için 
from scipy import stats

#min_max ölçeklendirm için
from mlxtend.preprocessing import minmax_scaling

# Çizdirme(plot) işlemleri için modülleri ekliyoruz
import seaborn as sns
import matplotlib.pyplot as plt

# Bu aşamada da veri setimizi okuyoruz
kickstarters_2017 = pd.read_csv ("../input/kickstarter-projects/ks-projects-201801.csv")

# Tekrar işlemleri için bu kodu yazıyoruz
np.random.seed(0)
# Ekponansiyel dağılımdan rastgele 100 tana veri noktası üretelim 
original_data = np.random.exponential(size = 100)

#min-max ölçeklendirme iin 0 ve 1 değerlerini kullanalım
scaled_data = minmax_scaling(original_data, columns = [0])

# şimdi de şekli çizdirerek kaşılaştıralım
fig, ax = plt.subplots(1, 2)
sns.distplot(original_data, ax = ax[0])
ax[0].set_title("Original Data")
sns.distplot(scaled_data, ax=ax[1])
ax[1].set_title("Scaled Data")
#Eksponansiyel dağılımdan 1000 adet veri notası oluşturuyoruz
original_data = np.random.exponential(size = 1000)

#Verileri 0 ve 1 arasında ölçeklendiriyoruz
scaled_data = minmax_scaling(original_data, columns = [0])

#Her ikisini birlikte karşılaştırmalı olarak çizdiriyoruz
fig, ax = plt.subplots(1, 2)
sns.distplot(original_data, ax = ax[0])
ax[0].set_title("Orjinal Veri")
sns.distplot(scaled_data, ax=ax[1])
ax[1].set_title("Ölçeklendirilmiş Veri")
#boxcox ile verilerin eksponansiyel oalrak normalize edilmesi
normalized_data = stats.boxcox(original_data)

#Şimdi her ikisinide karşılaştırmalı olarak çizdirelim
fig, ax = plt.subplots(1,2)
sns.distplot(original_data, ax = ax[0])
ax[0].set_title("Orijinal Veri")
sns.distplot(normalized_data[0], ax = ax[1])
ax[1].set_title("Normalize edilmiş Veri") 
#usd_goal_real sütünunu seçelim
usd_goal = kickstarters_2017.usd_goal_real

# Hedef değerleri 0 ile 1 değerlerine ölçekleyelim
scaled_data = minmax_scaling(usd_goal, columns = [0])

#Şimdi Orjinal veri ile Ölçeklendirilmiş veriyi çizdirelim
fig, ax = plt.subplots(1,2)
sns.distplot(kickstarters_2017.usd_goal_real, ax=ax[0])
ax[0].set_title("Orjinal Veri")
sns.distplot(scaled_data, ax=ax[1])
ax[1].set_title("Ölçeklendirilmiş Veri")

# Your turn! 

# We just scaled the "usd_goal_real" column. What about the "goal" column?
#usd_goal_real sütünunu seçelim
goal_1 = kickstarters_2017.goal

# Hedef değerleri 0 ile 1 değerlerine ölçekleyelim
scaled_data = minmax_scaling(goal_1, columns = [0])

#Şimdi Orjinal veri ile Ölçeklendirilmiş veriyi çizdirelim
fig, ax = plt.subplots(1,2)
sns.distplot(kickstarters_2017.goal, ax=ax[0])
ax[0].set_title("Orjinal Veri")
sns.distplot(scaled_data, ax=ax[1])
ax[1].set_title("Ölçeklendirilmiş Veri")
#Tüm olası pozitif durumların indexleyelim(Bunun için Box-Cox işlemini kullanacağız çünkü Box-Cox sadece pozitif değerleri alıyor)
index_of_positive_pledges = kickstarters_2017.usd_pledged_real > 0

# Sadece Pozitif teklifleri alacağız
pozitive_pledges = kickstarters_2017.usd_pledged_real.loc[index_of_positive_pledges]

#Vaat Edilmiş teklifleri Normalize edelim(Pledged)
normalized_pledges = stats.boxcox(positive_pledges)[0]

#Her iki durumuda çizdirelim
fig, ax=plt.subplots(1,2)
sns.distplot(positive_pledges, ax=ax[0])
ax[0].set_title("Orjinal Veri")
sns.distplot(normalized_pledges, ax=ax[1])
ax[1].set_title("Normalize Edilmiş Veri")