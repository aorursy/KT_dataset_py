import pandas as pd

import altair as alt
tam_gruplanmis = pd.read_csv('../input/turkeycovid19/emir_mertoglu_covid19_calismasi.csv', parse_dates=['Tarih'])



tr = tam_gruplanmis[tam_gruplanmis['Country/Region'] == 'Turkey']
grafik = alt.Chart(tr).mark_bar().encode(

    x='monthdate(Tarih):O',

).properties(

    width=500

)
mavi = alt.value('#536DFE')

kirmizi = alt.value('#D32F2F')



grafik.encode(y='Vakalar', color = mavi).properties(title='Toplam Vakalar') | grafik.encode(y='Olumler', color=kirmizi).properties(title='Toplam Olumler')
mavi = alt.value('#536DFE')

kirmizi = alt.value('#D32F2F')

yesil = alt.value('#388E3C')



grafik.encode(y='Yeni Vakalar', color = mavi).properties(title='Gunluk Yeni Vaka Sayisi') | grafik.encode(y='Yeni Olumler', color = kirmizi).properties(title='Gunluk Yeni Olum Sayisi') | grafik.encode(y='Yeni Iyilesenler', color = yesil).properties(title='Gunluk Yeni Iyilesme Sayisi')