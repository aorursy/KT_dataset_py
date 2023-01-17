import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
multiple = pd.read_csv("../input/multipleChoiceResponses.csv",dtype = np.object)
Hispano = multiple[(multiple.Q3 =="Mexico") | (multiple.Q3 =="Colombia") | (multiple.Q3 =="Argentina") | (multiple.Q3 =="Peru")\
                   | (multiple.Q3 =="Chile") ]
razon = float('{:03.1f}'.format(Hispano.Q3.count()/multiple.Q3.iloc[1:].count()*100))


Brazil =  multiple[multiple.Q3 =="Brazil"]
razon_Bra = float('{:03.1f}'.format(Brazil.Q3.count()/multiple.Q3.iloc[1:].count()*100))

Africa = multiple[(multiple.Q3 =="Nigeria") | (multiple.Q3 == 'South Africa') | (multiple.Q3 == 'Egypt') |\
                  (multiple.Q3 == 'Kenya') | (multiple.Q3 == 'Tunisia') | (multiple.Q3 == 'Morocco')]
razon_Af = float('{:03.1f}'.format(Africa.Q3.count()/multiple.Q3.iloc[1:].count()*100))

AsiaSE = multiple[(multiple.Q3 =="Singapore") | (multiple.Q3 == 'Indonesia') | (multiple.Q3 == 'Malaysia') \
                  | (multiple.Q3 == 'Philippines') | (multiple.Q3 == 'Thailand') | (multiple.Q3 == 'Viet Nam')]
razon_ASE = float('{:03.1f}'.format(AsiaSE.Q3.count()/multiple.Q3.iloc[1:].count()*100))

IndiaB = multiple[(multiple.Q3 =="India") | (multiple.Q3 == 'Bangladesh') | (multiple.Q3 == 'Pakistan')]
razon_IB = float('{:03.1f}'.format(IndiaB.Q3.count()/multiple.Q3.iloc[1:].count()*100))

ChinaH = multiple[(multiple.Q3 =="China") | (multiple.Q3 == 'Hong Kong (S.A.R.)')]
razon_CH = float('{:03.1f}'.format(ChinaH.Q3.count()/multiple.Q3.iloc[1:].count()*100))

RW = float('{:03.1f}'.format(100 -(razon + razon_Bra + razon_CH + razon_IB + razon_ASE + razon_Af)))

pd.DataFrame({'Region':["Hispano-America","Brazil","East Asia","South Asia","South East Asia","Africa",\
                        "Rest world"],'Participation in the survey (%)':[razon, razon_Bra, razon_CH, razon_IB,\
                                                                    razon_ASE, razon_Af, RW]})
country = Hispano.Q3.value_counts(normalize = True)*100
countryg = country.plot(kind = 'barh',figsize=(12,6), title = 'Hispanic American kagglers')

TCountHisp = []

for i in countryg.patches:
    TCountHisp.append(i.get_width())
    
tCountHisp = sum(TCountHisp)

for i in countryg.patches:
    countryg.text(i.get_width()+.3, i.get_y()+.38, \
            str(round((i.get_width()/tCountHisp)*100, 2))+'%', fontsize=8)

country_Af = Africa.Q3.value_counts(normalize = True)*100
country_Afg = country_Af.plot(kind = 'barh',figsize=(12,6), title = 'African kagglers')

TCountAfr = []

for i in country_Afg.patches:
    TCountAfr.append(i.get_width())
    
tCountAfr = sum(TCountAfr)

for i in country_Afg.patches:
    country_Afg.text(i.get_width()+.3, i.get_y()+.38, \
            str(round((i.get_width()/tCountAfr)*100, 2))+'%', fontsize=8)

country_ASE = AsiaSE.Q3.value_counts(normalize = True)*100
country_ASEg = country_ASE.plot(kind = 'barh',figsize=(12,6), title = 'Kagglers in Southeast Asia')

TCountASE = []

for i in country_ASEg.patches:
    TCountASE.append(i.get_width())
    
tCountASE = sum(TCountASE)

for i in country_ASEg.patches:
    country_ASEg.text(i.get_width()+.3, i.get_y()+.38, \
            str(round((i.get_width()/tCountASE)*100, 2))+'%', fontsize=8)



country_IB = IndiaB.Q3.value_counts(normalize = True)*100
country_IBg = country_IB.plot(kind = 'barh',figsize=(12,6), title = 'Kagglers in South Asia')

TCountIB = []

for i in country_IBg.patches:
    TCountIB.append(i.get_width())
    
tCountIB = sum(TCountIB)

for i in country_IBg.patches:
    country_IBg.text(i.get_width()+.3, i.get_y()+.38, \
            str(round((i.get_width()/tCountIB)*100, 2))+'%', fontsize=8)

country_CH = ChinaH.Q3.value_counts(normalize = True)*100
country_CHg = country_CH.plot(kind = 'barh',figsize=(12,6), title = 'Kagglers in East Asia')

TCountCH = []

for i in country_CHg.patches:
    TCountCH.append(i.get_width())
    
tCountCH = sum(TCountCH)

for i in country_CHg.patches:
    country_CHg.text(i.get_width()+.3, i.get_y()+.38, \
            str(round((i.get_width()/tCountCH)*100, 2))+'%', fontsize=8)

age_Hisp = Hispano.Q2.iloc[1:].value_counts(normalize = True)*100
age_Hispg = age_Hisp.plot(kind = 'barh' ,figsize=(12,6),title = "Hispanoamerica")

TAgeHisp = []

for i in age_Hispg.patches:
    TAgeHisp.append(i.get_width())
    
tAgeHisp = sum(TAgeHisp)

for i in age_Hispg.patches:
    age_Hispg.text(i.get_width()+.3, i.get_y()+.38, \
            str(round((i.get_width()/tAgeHisp)*100, 2))+'%', fontsize=8)

age_Bra = Brazil.Q2.iloc[1:].value_counts(normalize = True)*100
age_Brag = age_Bra.plot(kind = 'barh' ,figsize=(12,6),title = "Brazil")

TAgeBra = []

for i in age_Brag.patches:
    TAgeBra.append(i.get_width())
    
tAgeBra = sum(TAgeBra)

for i in age_Brag.patches:
    age_Brag.text(i.get_width()+.3, i.get_y()+.38, \
            str(round((i.get_width()/tAgeBra)*100, 2))+'%', fontsize=8)

age_Afr = Africa.Q2.iloc[1:].value_counts(normalize = True)*100
age_Afrg = age_Afr.plot(kind = 'barh' ,figsize=(12,6),title = "Africa")

TAgeAfr = []

for i in age_Afrg.patches:
    TAgeAfr.append(i.get_width())
    
tAgeAfr = sum(TAgeAfr)

for i in age_Afrg.patches:
    age_Afrg.text(i.get_width()+.3, i.get_y()+.38, \
            str(round((i.get_width()/tAgeAfr)*100, 2))+'%', fontsize=8)

age_ASE = AsiaSE.Q2.iloc[1:].value_counts(normalize = True)*100
age_ASEg = age_ASE.plot(kind = 'barh' ,figsize=(12,6),title = "South East Asia")

TAgeASE = []

for i in age_ASEg.patches:
    TAgeASE.append(i.get_width())
    
tAgeASE = sum(TAgeASE)

for i in age_ASEg.patches:
    age_ASEg.text(i.get_width()+.3, i.get_y()+.38, \
            str(round((i.get_width()/tAgeASE)*100, 2))+'%', fontsize=8)

age_IB = IndiaB.Q2.iloc[1:].value_counts(normalize = True)*100
age_IBg = age_IB.plot(kind = 'barh' ,figsize=(12,6),title = "South Asia")

TAgeIB = []

for i in age_IBg.patches:
    TAgeIB.append(i.get_width())
    
tAgeIB = sum(TAgeIB)

for i in age_IBg.patches:
    age_IBg.text(i.get_width()+.3, i.get_y()+.38, \
            str(round((i.get_width()/tAgeIB)*100, 2))+'%', fontsize=8)

age_ChinaH = ChinaH.Q2.iloc[1:].value_counts(normalize = True)*100
age_CHg = age_ChinaH.plot(kind = 'barh' ,figsize=(12,6),title = "Eastern Asia")

TAgeCH = []

for i in age_CHg.patches:
    TAgeCH.append(i.get_width())
    
tAgeCH = sum(TAgeCH)

for i in age_CHg.patches:
    age_CHg.text(i.get_width()+.3, i.get_y()+.38, \
            str(round((i.get_width()/tAgeCH)*100, 2))+'%', fontsize=8)

gender_Hisp = Hispano.Q1.iloc[1:].value_counts(normalize = True)*100
gender_Hispg = gender_Hisp.plot(kind = 'barh',figsize = (12,6),title = 'Hispanoamerica')

TGenHisp = []

for i in gender_Hispg.patches:
    TGenHisp.append(i.get_width())
    
tGenHisp = sum(TGenHisp)

for i in gender_Hispg.patches:
    gender_Hispg.text(i.get_width()+.3, i.get_y()+.38, \
            str(round((i.get_width()/tGenHisp)*100, 2))+'%', fontsize=8)
gender_Bra = Brazil.Q1.iloc[1:].value_counts(normalize = True)*100
gender_Brag = gender_Bra.plot(kind = 'barh',figsize = (12,6),title = 'Brazil')

TGenBra = []

for i in gender_Brag.patches:
    TGenBra.append(i.get_width())
    
tGenBra = sum(TGenBra)

for i in gender_Brag.patches:
    gender_Brag.text(i.get_width()+.3, i.get_y()+.38, \
            str(round((i.get_width()/tGenBra)*100, 2))+'%', fontsize=8)

gender_Afr = Africa.Q1.iloc[1:].value_counts(normalize = True)*100
gender_Afrg = gender_Afr.plot(kind = 'barh',figsize = (12,6),title = 'Africa')

TGenAfr = []

for i in gender_Afrg.patches:
    TGenAfr.append(i.get_width())
    
tGenAfr = sum(TGenAfr)

for i in gender_Afrg.patches:
    gender_Afrg.text(i.get_width()+.3, i.get_y()+.38, \
            str(round((i.get_width()/tGenAfr)*100, 2))+'%', fontsize=8)
gender_ASE = AsiaSE.Q1.iloc[1:].value_counts(normalize = True)*100
gender_ASEg = gender_ASE.plot(kind = 'barh',figsize = (12,6),title = 'South East Asia')

TGenASE = []

for i in gender_ASEg.patches:
    TGenASE.append(i.get_width())
    
tGenASE = sum(TGenASE)

for i in gender_ASEg.patches:
    gender_ASEg.text(i.get_width()+.3, i.get_y()+.38, \
            str(round((i.get_width()/tGenASE)*100, 2))+'%', fontsize=8)
gender_IB = IndiaB.Q1.iloc[1:].value_counts(normalize = True)*100
gender_IBg = gender_IB.plot(kind = 'barh',figsize = (12,6),title = 'South Asia')

TGenIB = []

for i in gender_IBg.patches:
    TGenIB.append(i.get_width())
    
tGenIB = sum(TGenIB)

for i in gender_IBg.patches:
    gender_IBg.text(i.get_width()+.3, i.get_y()+.38, \
            str(round((i.get_width()/tGenIB)*100, 2))+'%', fontsize=8)
gender_CH = ChinaH.Q1.iloc[1:].value_counts(normalize = True)*100
gender_CHg = gender_CH.plot(kind = 'barh',figsize = (12,6),title = 'Easter Asia')

TGenCH = []

for i in gender_CHg.patches:
    TGenCH.append(i.get_width())
    
tGenCH = sum(TGenCH)

for i in gender_CHg.patches:
    gender_CHg.text(i.get_width()+.3, i.get_y()+.38, \
            str(round((i.get_width()/tGenCH)*100, 2))+'%', fontsize=8)
education_Hisp = Hispano.Q4.iloc[1:].value_counts(normalize = True)*100
education_Hispg = education_Hisp.plot(kind = 'barh',figsize = (12,6), title = 'Hispanoamerica')

TEducHisp = []

for i in education_Hispg.patches:
    TEducHisp.append(i.get_width())
    
tEducHisp = sum(TEducHisp)

for i in education_Hispg.patches:
    education_Hispg.text(i.get_width()+.3, i.get_y()+.38, \
            str(round((i.get_width()/tEducHisp)*100, 2))+'%', fontsize=8)
education_Bra = Brazil.Q4.iloc[1:].value_counts(normalize = True)*100
education_Brag = education_Bra.plot(kind = 'barh',figsize = (12,6), title = 'Brazil')

TEducBra = []

for i in education_Brag.patches:
    TEducBra.append(i.get_width())
    
tEducBra = sum(TEducBra)

for i in education_Brag.patches:
    education_Brag.text(i.get_width()+.3, i.get_y()+.38, \
            str(round((i.get_width()/tEducBra)*100, 2))+'%', fontsize=8)
education_Afr = Africa.Q4.iloc[1:].value_counts(normalize = True)*100
education_Afrg = education_Afr.plot(kind = 'barh',figsize = (12,6), title = 'Africa')

TEducAfr = []

for i in education_Afrg.patches:
    TEducAfr.append(i.get_width())
    
tEducAfr = sum(TEducAfr)

for i in education_Afrg.patches:
    education_Afrg.text(i.get_width()+.3, i.get_y()+.38, \
            str(round((i.get_width()/tEducAfr)*100, 2))+'%', fontsize=8)
education_ASE = AsiaSE.Q4.iloc[1:].value_counts(normalize = True)*100
education_ASEg = education_ASE.plot(kind = 'barh',figsize = (12,6), title = 'South East Asia')
TEducASE = []

for i in education_ASEg.patches:
    TEducASE.append(i.get_width())
    
tEducASE = sum(TEducASE)

for i in education_ASEg.patches:
    education_ASEg.text(i.get_width()+.3, i.get_y()+.38, \
            str(round((i.get_width()/tEducASE)*100, 2))+'%', fontsize=8)
education_IB = IndiaB.Q4.iloc[1:].value_counts(normalize = True)*100
education_IBg = education_IB.plot(kind = 'barh',figsize = (12,6), title = 'South Asia')
TEducIB = []

for i in education_IBg.patches:
    TEducIB.append(i.get_width())
    
tEducIB = sum(TEducIB)

for i in education_IBg.patches:
    education_IBg.text(i.get_width()+.3, i.get_y()+.38, \
            str(round((i.get_width()/tEducIB)*100, 2))+'%', fontsize=8)
education_CH = ChinaH.Q4.iloc[1:].value_counts(normalize = True)*100
education_CHg = education_CH.plot(kind = 'barh',figsize = (12,6), title = 'Easter Asia')
TEducCH = []

for i in education_CHg.patches:
    TEducCH.append(i.get_width())
    
tEducCH = sum(TEducCH)

for i in education_CHg.patches:
    education_CHg.text(i.get_width()+.3, i.get_y()+.38, \
            str(round((i.get_width()/tEducCH)*100, 2))+'%', fontsize=8)
profession_Hisp = Hispano.Q5.iloc[1:].value_counts(normalize = True)*100
profession_Hispg = profession_Hisp.plot(kind = 'barh',figsize = (14,8), title = 'Hispanoamerica')

TProHisp = []

for i in profession_Hispg.patches:
    TProHisp.append(i.get_width())
    
tProHisp = sum(TProHisp)

for i in profession_Hispg.patches:
    profession_Hispg.text(i.get_width()+.3, i.get_y()+.38, \
            str(round((i.get_width()/tProHisp)*100, 2))+'%', fontsize=10)

profession_Bra = Brazil.Q5.iloc[1:].value_counts(normalize = True)*100
profession_Brag = profession_Bra.plot(kind = 'barh',figsize = (14,8), title = 'Brazil')

TProBra = []

for i in profession_Brag.patches:
    TProBra.append(i.get_width())
    
tProBra = sum(TProBra)

for i in profession_Brag.patches:
    profession_Brag.text(i.get_width()+.3, i.get_y()+.38, \
            str(round((i.get_width()/tProBra)*100, 2))+'%', fontsize=10)
profession_Afr = Africa.Q5.iloc[1:].value_counts(normalize = True)*100
profession_Afrg = profession_Afr.plot(kind = 'barh',figsize = (12,6), title = 'Africa')
TProAfr = []

for i in profession_Afrg.patches:
    TProAfr.append(i.get_width())
    
tProAfr = sum(TProAfr)

for i in profession_Afrg.patches:
    profession_Afrg.text(i.get_width()+.3, i.get_y()+.38, \
            str(round((i.get_width()/tProAfr)*100, 2))+'%', fontsize=10)
profession_ASE = AsiaSE.Q5.iloc[1:].value_counts(normalize = True)*100
profession_ASEg = profession_ASE.plot(kind = 'barh',figsize = (12,6), title = 'South East Asia')
TProASE = []

for i in profession_ASEg.patches:
    TProASE.append(i.get_width())
    
tProASE = sum(TProASE)

for i in profession_ASEg.patches:
    profession_ASEg.text(i.get_width()+.3, i.get_y()+.38, \
            str(round((i.get_width()/tProASE)*100, 2))+'%', fontsize=10)
profession_IB = IndiaB.Q5.iloc[1:].value_counts(normalize = True)*100
profession_IBg = profession_IB.plot(kind = 'barh',figsize = (12,6), title = 'South Asia')
TProIB = []

for i in profession_IBg.patches:
    TProIB.append(i.get_width())
    
tProIB = sum(TProIB)

for i in profession_IBg.patches:
    profession_IBg.text(i.get_width()+.3, i.get_y()+.38, \
            str(round((i.get_width()/tProIB)*100, 2))+'%', fontsize=10)
profession_CH = ChinaH.Q5.iloc[1:].value_counts(normalize = True)*100
profession_CHg = profession_CH.plot(kind = 'barh',figsize = (12,6), title = 'Easter Asia')
TProCH = []

for i in profession_CHg.patches:
    TProCH.append(i.get_width())
    
tProCH = sum(TProCH)

for i in profession_CHg.patches:
    profession_CHg.text(i.get_width()+.3, i.get_y()+.38, \
            str(round((i.get_width()/tProCH)*100, 2))+'%', fontsize=10)
role_Hisp = Hispano.Q6.iloc[1:].value_counts(normalize = True)*100
role_Hispg = role_Hisp.plot(kind = 'barh',figsize = (12,6),title = 'Hispanoamerica')

TRoleHisp = []

for i in role_Hispg.patches:
    TRoleHisp.append(i.get_width())
    
tRoleHisp = sum(TRoleHisp)

for i in role_Hispg.patches:
    role_Hispg.text(i.get_width()+.3, i.get_y()+.38, \
            str(round((i.get_width()/tRoleHisp)*100, 2))+'%', fontsize=10)
role_Bra = Brazil.Q6.iloc[1:].value_counts(normalize = True)*100
role_Brag = role_Bra.plot(kind = 'barh',figsize = (12,6),title = 'Brazil')

TRoleBra = []

for i in role_Brag.patches:
    TRoleBra.append(i.get_width())
    
tRoleBra = sum(TRoleBra)

for i in role_Brag.patches:
    role_Brag.text(i.get_width()+.3, i.get_y()+.38, \
            str(round((i.get_width()/tRoleBra)*100, 2))+'%', fontsize=10)
role_Afr = Africa.Q6.iloc[1:].value_counts(normalize = True)*100
role_Afrg = role_Afr.plot(kind = 'barh',figsize = (12,6),title = 'Africa')

TRoleAfr = []

for i in role_Afrg.patches:
    TRoleAfr.append(i.get_width())
    
tRoleAfr = sum(TRoleAfr)

for i in role_Afrg.patches:
    role_Afrg.text(i.get_width()+.3, i.get_y()+.38, \
            str(round((i.get_width()/tRoleAfr)*100, 2))+'%', fontsize=10)
role_ASE = AsiaSE.Q6.iloc[1:].value_counts(normalize = True)*100
role_ASEg = role_ASE.plot(kind = 'barh',figsize = (12,6),title = 'South East Asia')

TRoleASE = []

for i in role_ASEg.patches:
    TRoleASE.append(i.get_width())
    
tRoleASE = sum(TRoleASE)

for i in role_ASEg.patches:
    role_ASEg.text(i.get_width()+.3, i.get_y()+.38, \
            str(round((i.get_width()/tRoleASE)*100, 2))+'%', fontsize=10)
role_IB = IndiaB.Q6.iloc[1:].value_counts(normalize = True)*100
role_IBg = role_IB.plot(kind = 'barh',figsize = (12,6),title = 'South Asia')

TRoleIB = []

for i in role_IBg.patches:
    TRoleIB.append(i.get_width())
    
tRoleIB = sum(TRoleIB)

for i in role_IBg.patches:
    role_IBg.text(i.get_width()+.3, i.get_y()+.38, \
            str(round((i.get_width()/tRoleIB)*100, 2))+'%', fontsize=10)
role_CH = ChinaH.Q6.iloc[1:].value_counts(normalize = True)*100
role_CHg = role_CH.plot(kind = 'barh',figsize = (12,6),title = 'Africa')

TRoleCH = []

for i in role_CHg.patches:
    TRoleCH.append(i.get_width())
    
tRoleCH = sum(TRoleCH)

for i in role_CHg.patches:
    role_CHg.text(i.get_width()+.3, i.get_y()+.38, \
            str(round((i.get_width()/tRoleCH)*100, 2))+'%', fontsize=10)