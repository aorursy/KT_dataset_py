import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



from subprocess import check_output

#print(check_output(["ls", "../input"]).decode("utf8"))



%matplotlib inline

pd.set_option('display.max_rows', 10)



df = pd.read_csv("../input/2017.csv")



df.info()
#df.tail(3)
#df.plot(x='Happiness.Score', y='Economy..GDP.per.Capita.');
print("Väike ülevaade parimatest ja viletsamatest");

df[["Country","Happiness.Rank" ,"Freedom","Economy..GDP.per.Capita.","Generosity","Trust..Government.Corruption."]]





print("Keskmine õnnelikuse indeks:", df["Happiness.Score"].mean())

print("Maksimaalne õnnelikuse indeks:", df["Happiness.Score"].max())

print("Minimaalne õnnelikuse indeks:", df["Happiness.Score"].min())

print("Midagi " , df["Happiness.Score"].describe())
print("Kohustuslik groupby")

df.groupby("Happiness.Score")["Happiness.Score"].count()

print("Vaatame üldist õnnelikkuse jaotust");

print(df["Happiness.Score"].round(1).value_counts());

#df.Family.plot.hist()
print("Sama asi histogrammina on muidugi palju ilusam");

df["Happiness.Score"].plot.hist();
print("Tundub, et vabad inimesed on õnnelikumad");

df.plot.scatter("Freedom", "Happiness.Rank", alpha=0.2, color="Red");
print("Ka rikkamad inimesed on õnnelikumad, samas ei ole koondumine nii märgatav kui vabaduse puhul");

df.plot.scatter("Economy..GDP.per.Capita.", "Happiness.Rank", alpha=0.2, color="Red");
print("Uurime kuidas vaba tegutsemine ja rikkus seotud on");

df.plot.scatter("Economy..GDP.per.Capita.", "Freedom", alpha=0.2, color="Red");

print("Lahkus vs õnnelikkus ei näi sügavat seost omavat");

df.plot.scatter("Generosity", "Happiness.Rank", alpha=0.2, color="Red");
print("Küll aga näha, et vabamad on (natukenegi) lahkemad.");

df.plot.scatter("Generosity", "Freedom", alpha=0.2, color="Red");
print("Tundub, et õnnelikemaates maades usaldatakse valitsust natuke rohkem kuid mitte märkimisväärselt.");

df.plot(x='Happiness.Score', y='Trust..Government.Corruption.');