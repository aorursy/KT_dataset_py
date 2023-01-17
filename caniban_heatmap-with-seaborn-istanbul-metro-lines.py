import numpy as np

 

import seaborn as sb

 

import matplotlib.pyplot as plt



# The array of monthly trip frequencies of Istanbul Metro M1 Line (2017-2019)

tripm1 = np.array([[21293,19458,21581,21043,21565,20778,20514,21499,20773,21539,21029,21556],

 [21556,19600,21563,20862,21564,20791,21410,21421,20788,21575,20847,21555],

[21575,19488,21568,20871,21573,20799,21412,21421,21039,22064,21591,22157]])





# Adding months 

text = np.asarray([['2017 Jan', '2017 Feb', '2017 Mar', '2017 Apr', '2017 May', '2017 Jun', '2017 Jul', '2017 Aug', '2017 Sep', '2017 Oct', '2017 Nov', '2017 Dec'],

                   ['2018 Jan', '2018 Feb', '2018 Mar', '2018 Apr', '2018 May', '2018 Jun', '2018 Jul', '2018 Aug', '2018 Sep', '2018 Oct', '2018 Nov', '2018 Dec'],

                   ['2019 Jan', '2019 Feb', '2019 Mar', '2019 Apr', '2019 May', '2019 Jun', '2019 Jul', '2019 Aug', '2019 Sep', '2019 Oct', '2019 Nov', '2019 Dec']])



# Adding labels

labels = (np.asarray(["{0}\n{1:.0f}".format(text,tripm1) for text, tripm1 in zip(text.flatten(), tripm1.flatten())])).reshape(3,12)



# Setting the heatmap

sb.set(font_scale=1.2,rc={'figure.figsize':(20,10)})

ax=heat_map = sb.heatmap(tripm1, cmap="OrRd", annot=labels, fmt='',cbar_kws={'orientation': 'horizontal'}, yticklabels=False, xticklabels=False )

ax.figure.axes[-1].set_ylabel('Trip Frequency', size=15)

ax.set_title('Istanbul M1 Metro Line [Yenikapı - Kirazlı - Atatürk Airport] Monthly Trip Frequencies', fontsize = 30, weight='bold')

plt.show()

# The array of monthly trip frequencies of Istanbul Metro M2 Line (2017-2019)

tripm2 = np.array([[22471,20274,22835,21739,22587,22984,21782,22493,22324,24357,23695,24238],

                    [24354,22130,24275,24376,24830,23574,22627,19892,20410,21096,21188,21071],

                    [21465,19418,21540,20824,21873,20367,20785,20927,21636,21764,21280,21885]])





# Adding months 

text = np.asarray([['2017 Jan', '2017 Feb', '2017 Mar', '2017 Apr', '2017 May', '2017 Jun', '2017 Jul', '2017 Aug', '2017 Sep', '2017 Oct', '2017 Nov', '2017 Dec'],

                   ['2018 Jan', '2018 Feb', '2018 Mar', '2018 Apr', '2018 May', '2018 Jun', '2018 Jul', '2018 Aug', '2018 Sep', '2018 Oct', '2018 Nov', '2018 Dec'],

                   ['2019 Jan', '2019 Feb', '2019 Mar', '2019 Apr', '2019 May', '2019 Jun', '2019 Jul', '2019 Aug', '2019 Sep', '2019 Oct', '2019 Nov', '2019 Dec']])



# Adding labels

labels = (np.asarray(["{0}\n{1:.0f}".format(text,tripm2) for text, tripm2 in zip(text.flatten(), tripm2.flatten())])).reshape(3,12)



# Setting the heatmap

sb.set(font_scale=1.2,rc={'figure.figsize':(20,10)})

ax=heat_map = sb.heatmap(tripm2, cmap="YlGn", annot=labels, fmt='',cbar_kws={'orientation': 'horizontal'}, yticklabels=False, xticklabels=False )

ax.figure.axes[-1].set_ylabel('Trip Frequency', size=15)

ax.set_title('Istanbul M2 Metro Line [Yenikapı - Hacıosman] Monthly Trip Frequencies', fontsize = 30, weight='bold')

plt.show()
# The array of monthly trip frequencies of Istanbul Metro M3 Line (2017-2019)

tripm3 = np.array([[16557,15008,16682,16008,16559,15945,15929,16099,15707,16564,16133,16528],

[16564,15011,16653,15969,15712,16062,15877,15830,15662,16653,16130,16528],

[16565,15012,16529,16094,16653,15613,15893,15925,15969,16862,16298,16756]])





# Adding months (in Turkish)

text = np.asarray([['2017 Jan', '2017 Feb', '2017 Mar', '2017 Apr', '2017 May', '2017 Jun', '2017 Jul', '2017 Aug', '2017 Sep', '2017 Oct', '2017 Nov', '2017 Dec'],

                   ['2018 Jan', '2018 Feb', '2018 Mar', '2018 Apr', '2018 May', '2018 Jun', '2018 Jul', '2018 Aug', '2018 Sep', '2018 Oct', '2018 Nov', '2018 Dec'],

                   ['2019 Jan', '2019 Feb', '2019 Mar', '2019 Apr', '2019 May', '2019 Jun', '2019 Jul', '2019 Aug', '2019 Sep', '2019 Oct', '2019 Nov', '2019 Dec']])



# Adding labels

labels = (np.asarray(["{0}\n{1:.0f}".format(text,tripm3) for text, tripm3 in zip(text.flatten(), tripm3.flatten())])).reshape(3,12)



# Setting the heatmap

sb.set(font_scale=1.2,rc={'figure.figsize':(20,10)})

ax=heat_map = sb.heatmap(tripm3, cmap="Blues", annot=labels, fmt='',cbar_kws={'orientation': 'horizontal'}, yticklabels=False, xticklabels=False )

ax.figure.axes[-1].set_ylabel('Trip Frequency', size=15)

ax.set_title('Istanbul M3 Metro Line [Kirazlı - Olimpiyat - Başakşehir] Monthly Trip Frequencies', fontsize = 30, weight='bold')

plt.show()
# The array of monthly trip frequencies of Istanbul Metro M4 Line (2017-2019)

tripm4 = np.array([[11762,10626,11815,11412,11783,11200,11394,11482,11297,11801,11586,11848],

[11882,10708,11878,11445,12024,11181,10980,11065,10964,11729,11351,11667],

[11716,10576,11649,11328,11900,11114,11021,11287,11361,11998,11608,11988]])





# Adding months

text = np.asarray([['2017 Jan', '2017 Feb', '2017 Mar', '2017 Apr', '2017 May', '2017 Jun', '2017 Jul', '2017 Aug', '2017 Sep', '2017 Oct', '2017 Nov', '2017 Dec'],

                   ['2018 Jan', '2018 Feb', '2018 Mar', '2018 Apr', '2018 May', '2018 Jun', '2018 Jul', '2018 Aug', '2018 Sep', '2018 Oct', '2018 Nov', '2018 Dec'],

                   ['2019 Jan', '2019 Feb', '2019 Mar', '2019 Apr', '2019 May', '2019 Jun', '2019 Jul', '2019 Aug', '2019 Sep', '2019 Oct', '2019 Nov', '2019 Dec']])



# Adding labels

labels = (np.asarray(["{0}\n{1:.0f}".format(text,tripm4) for text, tripm4 in zip(text.flatten(), tripm4.flatten())])).reshape(3,12)



# Setting the heatmap

sb.set(font_scale=1.2,rc={'figure.figsize':(20,10)})

ax=heat_map = sb.heatmap(tripm4, cmap="PuRd", annot=labels, fmt='',cbar_kws={'orientation': 'horizontal'}, yticklabels=False, xticklabels=False )

ax.figure.axes[-1].set_ylabel('Trip Frequencies', size=15)

ax.set_title('Istanbul M4 Metro Line [Kadıköy - Tavşantepe] Monthly Trip Frequencies', fontsize = 30, weight='bold')

plt.show()
# The array of monthly trip frequencies of Istanbul Metro M6 Line (2017-2019)



tripm6 = np.array([[7499,6768,7648,7274,7515,7051,7502,7543,7260,7501,7258,7502],

[7500,6655,7438,7260,7678,7400,7530,7260,7256,7260,7018,7502],

[7502,6776,7491,7224,7762,7307,7502,7592,8021,9530,9343,9371]])





# Adding months

text = np.asarray([['2017 Jan', '2017 Feb', '2017 Mar', '2017 Apr', '2017 May', '2017 Jun', '2017 Jul', '2017 Aug', '2017 Sep', '2017 Oct', '2017 Nov', '2017 Dec'],

                   ['2018 Jan', '2018 Feb', '2018 Mar', '2018 Apr', '2018 May', '2018 Jun', '2018 Jul', '2018 Aug', '2018 Sep', '2018 Oct', '2018 Nov', '2018 Dec'],

                   ['2019 Jan', '2019 Feb', '2019 Mar', '2019 Apr', '2019 May', '2019 Jun', '2019 Jul', '2019 Aug', '2019 Sep', '2019 Oct', '2019 Nov', '2019 Dec']])





# Adding labels

labels = (np.asarray(["{0}\n{1:.0f}".format(text,tripm6) for text, tripm6 in zip(text.flatten(), tripm6.flatten())])).reshape(3,12)



# Setting the heatmap



sb.set(font_scale=1.2,rc={'figure.figsize':(20,10)})

ax=heat_map = sb.heatmap(tripm6, cmap="YlOrBr", annot=labels, fmt='',cbar_kws={'orientation': 'horizontal'}, yticklabels=False, xticklabels=False )

ax.figure.axes[-1].set_ylabel('Trip Frequencies', size=15)

ax.set_title('Istanbul M6 Metro Line [Levent - Boğaziçi University] Monthly Trip Frequencies', fontsize = 30, weight='bold')

plt.show()

# The array of monthly trip frequencies of Istanbul Metro M5 Line (2018-2019)

# This line was opened by 2018. There's no data for 2017.

tripm5 = np.array([[8247,7426,8179,8001,8408,8148,5678,8429,7067,9998,10183,10460],

[10908,9838,10462,10201,10537,10044,10242,10946,10693,11183,10989,10979]]) 



# Adding months

text = np.asarray([ ['2018 Jan', '2018 Feb', '2018 Mar', '2018 Apr', '2018 May', '2018 Jun', '2018 Jul', '2018 Aug', '2018 Sep', '2018 Oct', '2018 Nov', '2018 Dec'],

                   ['2019 Jan', '2019 Feb', '2019 Mar', '2019 Apr', '2019 May', '2019 Jun', '2019 Jul', '2019 Aug', '2019 Sep', '2019 Oct', '2019 Nov', '2019 Dec']])



# Adding labels

labels = (np.asarray(["{0}\n{1:.0f}".format(text,tripm5) for text, tripm5 in zip(text.flatten(), tripm5.flatten())])).reshape(2,12)



# Setting the heatmap

sb.set(font_scale=1.2,rc={'figure.figsize':(20,10)})

ax=heat_map = sb.heatmap(tripm5, cmap="RdPu", annot=labels, fmt='',cbar_kws={'orientation': 'horizontal'}, yticklabels=False, xticklabels=False )

ax.figure.axes[-1].set_ylabel('Trip Frequencies', size=15)

ax.set_title('Istanbul M5 Metro Line [Üsküdar - Ümraniye - Çekmeköy] Monthly Trip Frequencies', fontsize = 30, weight='bold')

plt.show()
