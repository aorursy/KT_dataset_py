import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style="whitegrid")

types = ["spell", "item", "hero", "artifact", "imba"]
limits = [5, 4, 3, 2, 1] #массив ограничений. Первый элемент - ограничение на карты заклинания, второй на предметы и т.д.
weights = [10, 5, 4, 2, 1] #массив весов (может не суммироваться в единицу)
n_draws = 10 #колчиество наград
n_trials = 1000 #количество итераций
def sample_multidraw_with_limits(p, limits, n_draws):
    assert len(p) == len(limits)
    assert sum(limits) > n_draws
    res = [0]*len(p)
    cnt = 0
    while(cnt < n_draws): 
        draw = np.random.multinomial(1, p).tolist()
        i = draw.index(1)
        if (res[i] + 1 <= limits[i]):
            res[i] += 1
            cnt += 1
    return res

p = np.array([i/sum(weights) for i in weights]) #переход от весов к вероятностям
draws = np.zeros((n_trials, len(p)))
for i in range(n_trials): 
    draws[i] = sample_multidraw_with_limits(p, limits, n_draws, )
draws_df = pd.DataFrame(draws, columns = types)

m_series = draws_df.mean()

ax_limits = sns.barplot(x = m_series.index, y = limits, palette = 'pastel' )
ax = sns.barplot(x = m_series.index, y = m_series, palette = 'bright')
ax.set_title('Матожидание количества копий')
ax.set(ylim=(0, max(limits)))
for i in range(m_series.count()):
    ax.text(i,m_series[i] + 0.05, m_series[i], color='black', ha="center") 
    
fig, axs = plt.subplots(ncols=5)
fig.suptitle('Гистограмы распределия количества копий (по типу)')
fig.set_size_inches((3*fig.get_size_inches()[0], fig.get_size_inches()[1]))
for i in range(len(limits)):
    sns.countplot(draws_df[types[i]], ax=axs[i], order = range(len(limits)+1))
