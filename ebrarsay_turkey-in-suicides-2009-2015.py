# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
dead=pd.read_csv("../input/suicide-rates-overview-1985-to-2016/master.csv")
dead.country.unique()
turkey=dead[dead.country=="Turkey"]

turkey.year.unique()
two_thousand_nine=turkey[turkey.year==2009]

two_thousand_ten=turkey[turkey.year==2010]

two_thousand_eleven=turkey[turkey.year==2011]

two_thousand_twelve=turkey[turkey.year==2012]

two_thousand_thirteen=turkey[turkey.year==2013]

two_thousand_fourtheen=turkey[turkey.year==2014]

two_thousand_fivetheen=turkey[turkey.year==2015]
male_2009=two_thousand_nine[two_thousand_nine.sex=="male"]

female_2009=two_thousand_nine[two_thousand_nine.sex=="female"]



male_2010=two_thousand_ten[two_thousand_ten.sex=="male"]

female_2010=two_thousand_ten[two_thousand_ten.sex=="female"]



male_2011=two_thousand_eleven[two_thousand_eleven.sex=="male"]

female_2011=two_thousand_eleven[two_thousand_eleven.sex=="female"]



male_2012=two_thousand_twelve[two_thousand_twelve.sex=="male"]

female_2012=two_thousand_twelve[two_thousand_twelve.sex=="female"]





male_2013=two_thousand_thirteen[two_thousand_thirteen.sex=="male"]

female_2013=two_thousand_thirteen[two_thousand_thirteen.sex=="female"]



male_2014=two_thousand_fourtheen[two_thousand_fourtheen.sex=="male"]

female_2014=two_thousand_fourtheen[two_thousand_fourtheen.sex=="female"]



male_2015=two_thousand_fivetheen[two_thousand_fivetheen.sex=="male"]

female_2015=two_thousand_fivetheen[two_thousand_fivetheen.sex=="female"]
2009



female_2009.plot(kind="line",x="age",y="population",color="Blue",alpha=0.5,grid=1)

plt.title("2009 YEARS FEMALE POPULATION-AGE GRAPHIC")

plt.xlabel("AGE")

plt.ylabel("POPULATION")

plt.show()



male_2009.plot(kind="line",x="age",y="population",color="Blue",alpha=0.5,grid=1)

plt.title("2009 YEARS MALE POPULATION-AGE GRAPHIC")

plt.xlabel("AGE")

plt.ylabel("POPULATION")

plt.show()

2010



female_2010.plot(kind="line",x="age",y="population",color="Red",alpha=0.5,grid=1)

plt.title("2010 YEARS FEMALE POPULATION-AGE GRAPHIC")

plt.xlabel("AGE")

plt.ylabel("POPULATION")

plt.show()



male_2010.plot(kind="line",x="age",y="population",color="Red",alpha=0.5,grid=1)

plt.title("2010 YEARS MALE POPULATION-AGE GRAPHIC")

plt.xlabel("AGE")

plt.ylabel("POPULATION")

plt.show()
2011



female_2011.plot(kind="line",x="age",y="population",color="Green",alpha=0.5,grid=1)

plt.title("2011 YEARS FEMALE POPULATION-AGE GRAPHIC")

plt.xlabel("AGE")

plt.ylabel("POPULATION")

plt.show()



male_2011.plot(kind="line",x="age",y="population",color="Green",alpha=0.5,grid=1)

plt.title("2011 YEARS MALE POPULATION-AGE GRAPHIC")

plt.xlabel("AGE")

plt.ylabel("POPULATION")

plt.show()
2012



female_2012.plot(kind="line",x="age",y="population",color="Blue",alpha=0.5,grid=1)

plt.title("2012 YEARS FEMALE POPULATION-AGE GRAPHIC")

plt.xlabel("AGE")

plt.ylabel("POPULATION")

plt.show()



male_2012.plot(kind="line",x="age",y="population",color="Blue",alpha=0.5,grid=1)

plt.title("2012 YEARS MALE POPULATION-AGE GRAPHIC")

plt.xlabel("AGE")

plt.ylabel("POPULATION")

plt.show()
2013



female_2013.plot(kind="line",x="age",y="population",color="Green",alpha=0.5,grid=1)

plt.title("2013 YEARS FEMALE POPULATION-AGE GRAPHIC")

plt.xlabel("AGE")

plt.ylabel("POPULATION")

plt.show()



male_2013.plot(kind="line",x="age",y="population",color="Green",alpha=0.5,grid=1)

plt.title("2013 YEARS MALE POPULATION-AGE GRAPHIC")

plt.xlabel("AGE")

plt.ylabel("POPULATION")

plt.show()
2014



female_2014.plot(kind="line",x="age",y="population",color="Red",alpha=0.5,grid=1)

plt.title("2014 YEARS FEMALE POPULATION-AGE GRAPHIC")

plt.xlabel("AGE")

plt.ylabel("POPULATION")

plt.show()



male_2014.plot(kind="line",x="age",y="population",color="Red",alpha=0.5,grid=1)

plt.title("2014 YEARS MALE POPULATION-AGE GRAPHIC")

plt.xlabel("AGE")

plt.ylabel("POPULATION")

plt.show()
2015



female_2015.plot(kind="line",x="age",y="population",color="Black",alpha=0.5,grid=1)

plt.title("2015 YEARS FEMALE POPULATION-AGE GRAPHIC")

plt.xlabel("AGE")

plt.ylabel("POPULATION")

plt.show()



male_2015.plot(kind="line",x="age",y="population",color="Black",alpha=0.5,grid=1)

plt.title("2015 YEARS MALE POPULATION-AGE GRAPHIC")

plt.xlabel("AGE")

plt.ylabel("POPULATION")

plt.show()
#2009 FEMALE



plt.subplot(5,1,1)

plt.plot(female_2009.age,female_2009.population,color="Red")

plt.title("2009 YEARS FEMALE POPULATION-AGE GRAPHIC")

plt.xlabel("AGE")

plt.ylabel("POPULATION")

plt.show()



plt.subplot(5,1,3)

plt.plot(female_2009.age,female_2009.suicides_no,color="Blue")

plt.title("2009 YEARS FEMALE SUICIDES-AGE GRAPHIC")

plt.xlabel("AGE")

plt.ylabel("POPULATION")

plt.show()



plt.subplot(5,1,5)

plt.plot(female_2009.age,(((female_2009.suicides_no)/(female_2009.population))*100000),color="Black")

plt.title("A SUICIDE RATE OF 100000 PEOPLE IN THE FEMALE POPULATION IN 2009")

plt.show()



#2009 MALE

plt.subplot(5,1,1)

plt.plot(male_2009.age,male_2009.population,color="Green")

plt.title("2009 YEARS MALE POPULATION-AGE GRAPHIC")

plt.xlabel("AGE")

plt.ylabel("POPULATION")

plt.show()



plt.subplot(5,1,3)

plt.plot(male_2009.age,male_2009.suicides_no,color="Orange")

plt.title("2009 YEARS MALE SUICIDES-AGE GRAPHIC")

plt.xlabel("AGE")

plt.ylabel("SUICIDES")

plt.show()



plt.subplot(5,1,5)

plt.plot(male_2009.age,(((male_2009.suicides_no)/(male_2009.population))*100000),color="purple")

plt.title("A SUICIDE RATE OF 100000 PEOPLE IN THE FEMALE POPULATION IN 2009")

plt.show()
#2010 FEMALE



plt.subplot(5,1,1)

plt.plot(female_2010.age,female_2010.population,color="Red")

plt.title("2010 YEARS FEMALE POPULATION-AGE GRAPHIC")

plt.xlabel("AGE")

plt.ylabel("POPULATION")

plt.show()



plt.subplot(5,1,3)

plt.plot(female_2010.age,female_2010.suicides_no,color="Blue")

plt.title("2010 YEARS FEMALE SUICIDES-AGE GRAPHIC")

plt.xlabel("AGE")

plt.ylabel("POPULATION")

plt.show()



plt.subplot(5,1,5)

plt.plot(female_2010.age,(((female_2010.suicides_no)/(female_2010.population))*100000),color="Black")

plt.title("A SUICIDE RATE OF 100000 PEOPLE IN THE FEMALE POPULATION IN 2010")

plt.show()



#2010 MALE



plt.subplot(5,1,1)

plt.plot(male_2010.age,male_2010.population,color="Green")

plt.title("2010 YEARS MALE POPULATION-AGE GRAPHIC")

plt.xlabel("AGE")

plt.ylabel("POPULATION")

plt.show()



plt.subplot(5,1,3)

plt.plot(male_2010.age,male_2010.suicides_no,color="Orange")

plt.title("2010 YEARS MALE SUICIDES-AGE GRAPHIC")

plt.xlabel("AGE")

plt.ylabel("SUICIDES")

plt.show()



plt.subplot(5,1,5)

plt.plot(male_2010.age,(((male_2010.suicides_no)/(male_2010.population))*100000),color="purple")

plt.title("A SUICIDE RATE OF 100000 PEOPLE IN THE FEMALE POPULATION IN 2010")

plt.show()
#2011 FEMALE



plt.subplot(5,1,1)

plt.plot(female_2011.age,female_2011.population,color="Red")

plt.title("2011 YEARS FEMALE POPULATION-AGE GRAPHIC")

plt.xlabel("AGE")

plt.ylabel("POPULATION")

plt.show()



plt.subplot(5,1,3)

plt.plot(female_2011.age,female_2011.suicides_no,color="Blue")

plt.title("2011 YEARS FEMALE SUICIDES-AGE GRAPHIC")

plt.xlabel("AGE")

plt.ylabel("POPULATION")

plt.show()



plt.subplot(5,1,5)

plt.plot(female_2011.age,(((female_2011.suicides_no)/(female_2011.population))*100000),color="Black")

plt.title("A SUICIDE RATE OF 100000 PEOPLE IN THE FEMALE POPULATION IN 2011")

plt.show()



#2011 MALE



plt.subplot(5,1,1)

plt.plot(male_2011.age,male_2011.population,color="green")

plt.title("2011 YEARS MALE POPULATION-AGE GRAPHIC")

plt.xlabel("AGE")

plt.ylabel("POPULATION")

plt.show()



plt.subplot(5,1,3)

plt.plot(male_2011.age,male_2011.suicides_no,color="orange")

plt.title("2011 YEARS MALE SUICIDES-AGE GRAPHIC")

plt.xlabel("AGE")

plt.ylabel("SUICIDES")

plt.show()



plt.subplot(5,1,5)

plt.plot(male_2011.age,(((male_2011.suicides_no)/(male_2011.population))*100000),color="purple")

plt.title("A SUICIDE RATE OF 100000 PEOPLE IN THE FEMALE POPULATION IN 2011")

plt.show()
#2012 FEMALE



plt.subplot(5,1,1)

plt.plot(female_2012.age,female_2012.population,color="Red")

plt.title("2012 YEARS FEMALE POPULATION-AGE GRAPHIC")

plt.xlabel("AGE")

plt.ylabel("POPULATION")

plt.show()



plt.subplot(5,1,3)

plt.plot(female_2012.age,female_2012.suicides_no,color="Blue")

plt.title("2012 YEARS FEMALE SUICIDES-AGE GRAPHIC")

plt.xlabel("AGE")

plt.ylabel("POPULATION")

plt.show()



plt.subplot(5,1,5)

plt.plot(female_2012.age,(((female_2012.suicides_no)/(female_2012.population))*100000),color="Black")

plt.title("A SUICIDE RATE OF 100000 PEOPLE IN THE FEMALE POPULATION IN 2012")

plt.show()



#2012 MALE



plt.subplot(5,1,1)

plt.plot(male_2012.age,male_2012.population,color="green")

plt.title("2012 YEARS MALE POPULATION-AGE GRAPHIC")

plt.xlabel("AGE")

plt.ylabel("POPULATION")

plt.show()



plt.subplot(5,1,3)

plt.plot(male_2012.age,male_2012.suicides_no,color="orange")

plt.title("2012 YEARS MALE SUICIDES-AGE GRAPHIC")

plt.xlabel("AGE")

plt.ylabel("SUICIDES")

plt.show()



plt.subplot(5,1,5)

plt.plot(male_2012.age,(((male_2012.suicides_no)/(male_2012.population))*100000),color="purple")

plt.title("A SUICIDE RATE OF 100000 PEOPLE IN THE FEMALE POPULATION IN 2012")

plt.show()
#2013 FEMALE



plt.subplot(5,1,1)

plt.plot(female_2013.age,female_2013.population,color="Red")

plt.title("2013 YEARS FEMALE POPULATION-AGE GRAPHIC")

plt.xlabel("AGE")

plt.ylabel("POPULATION")

plt.show()



plt.subplot(5,1,3)

plt.plot(female_2013.age,female_2013.suicides_no,color="Blue")

plt.title("2013 YEARS FEMALE SUICIDES-AGE GRAPHIC")

plt.xlabel("AGE")

plt.ylabel("POPULATION")

plt.show()



plt.subplot(5,1,5)

plt.plot(female_2013.age,(((female_2013.suicides_no)/(female_2013.population))*100000),color="Black")

plt.title("A SUICIDE RATE OF 100000 PEOPLE IN THE FEMALE POPULATION IN 2013")

plt.show()





#2013 MALE



plt.subplot(5,1,1)

plt.plot(male_2013.age,male_2013.population,color="green")

plt.title("2013 YEARS MALE POPULATION-AGE GRAPHIC")

plt.xlabel("AGE")

plt.ylabel("POPULATION")

plt.show()



plt.subplot(5,1,3)

plt.plot(male_2013.age,male_2013.suicides_no,color="orange")

plt.title("2013 YEARS MALE SUICIDES-AGE GRAPHIC")

plt.xlabel("AGE")

plt.ylabel("SUICIDES")

plt.show()



plt.subplot(5,1,5)

plt.plot(male_2013.age,(((male_2013.suicides_no)/(male_2013.population))*100000),color="purple")

plt.title("A SUICIDE RATE OF 100000 PEOPLE IN THE FEMALE POPULATION IN 2013")

plt.show()
#2014 FEMALE



plt.subplot(5,1,1)

plt.plot(female_2014.age,female_2014.population,color="Red")

plt.title("2014 YEARS FEMALE POPULATION-AGE GRAPHIC")

plt.xlabel("AGE")

plt.ylabel("POPULATION")

plt.show()



plt.subplot(5,1,3)

plt.plot(female_2014.age,female_2014.suicides_no,color="Blue")

plt.title("2014 YEARS FEMALE SUICIDES-AGE GRAPHIC")

plt.xlabel("AGE")

plt.ylabel("POPULATION")

plt.show()



plt.subplot(5,1,5)

plt.plot(female_2014.age,(((female_2014.suicides_no)/(female_2014.population))*100000),color="Black")

plt.title("A SUICIDE RATE OF 100000 PEOPLE IN THE FEMALE POPULATION IN 2014")

plt.show()



#2014 MALE



plt.subplot(5,1,1)

plt.plot(male_2014.age,male_2014.population,color="green")

plt.title("2014 YEARS MALE POPULATION-AGE GRAPHIC")

plt.xlabel("AGE")

plt.ylabel("POPULATION")

plt.show()



plt.subplot(5,1,3)

plt.plot(male_2014.age,male_2014.suicides_no,color="orange")

plt.title("2014 YEARS MALE SUICIDES-AGE GRAPHIC")

plt.xlabel("AGE")

plt.ylabel("SUICIDES")

plt.show()



plt.subplot(5,1,5)

plt.plot(male_2014.age,(((male_2014.suicides_no)/(male_2014.population))*100000),color="purple")

plt.title("A SUICIDE RATE OF 100000 PEOPLE IN THE FEMALE POPULATION IN 2014")

plt.show()
#2015 FEMALE



plt.subplot(5,1,1)

plt.plot(female_2015.age,female_2015.population,color="Red")

plt.title("2015 YEARS FEMALE POPULATION-AGE GRAPHIC")

plt.xlabel("AGE")

plt.ylabel("POPULATION")

plt.show()



plt.subplot(5,1,3)

plt.plot(female_2015.age,female_2015.suicides_no,color="Blue")

plt.title("2015 YEARS FEMALE SUICIDES-AGE GRAPHIC")

plt.xlabel("AGE")

plt.ylabel("POPULATION")

plt.show()



plt.subplot(5,1,5)

plt.plot(female_2015.age,(((female_2015.suicides_no)/(female_2015.population))*100000),color="Black")

plt.title("A SUICIDE RATE OF 100000 PEOPLE IN THE FEMALE POPULATION IN 2015")



plt.show()



#2015 MALE



plt.subplot(5,1,1)

plt.plot(male_2015.age,male_2015.population,color="green")

plt.title("2015 YEARS MALE POPULATION-AGE GRAPHIC")

plt.xlabel("AGE")

plt.ylabel("POPULATION")

plt.show()



plt.subplot(5,1,3)

plt.plot(male_2015.age,male_2015.suicides_no,color="orange")

plt.title("2015 YEARS MALE SUICIDES-AGE GRAPHIC")

plt.xlabel("AGE")

plt.ylabel("SUICIDES")

plt.show()



plt.subplot(5,1,5)

plt.plot(male_2015.age,(((male_2015.suicides_no)/(male_2015.population))*100000),color="purple")

plt.title("A SUICIDE RATE OF 100000 PEOPLE IN THE FEMALE POPULATION IN 2015")

plt.show()