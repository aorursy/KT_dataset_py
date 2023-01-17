import os
os.listdir('../input/')
plik_pensje = open('../input/pensje.csv', 'r')
podstawy = dict()
prowizje = dict()
plik_pensje.readline()

for line in plik_pensje:
    imie     = line.strip().split(',')[0].strip().lower()
    nazwisko = line.strip().split(',')[1].strip().lower()
    podstawa = line.strip().split(',')[2]
    prowizja = line.strip().split(',')[3]
    imie_nazwisko = imie + ' ' + nazwisko
    podstawy[imie_nazwisko] = int(podstawa)
    prowizje[imie_nazwisko] = float(prowizja)
print(podstawy)
print(prowizje)

print('')

plik_umowy = open('../input/umowy.csv', 'r')

plik_umowy.readline()

czasopisma_osoby = dict()
ile_razy = dict()

for line in plik_umowy:
    podzielone = line.strip().split(',')
    czasopismo = podzielone[0].strip().lower()
    imie_nazwisko = podzielone[2].strip().lower()
    rok = int(podzielone[1].strip())
    czasopisma_osoby[czasopismo] = imie_nazwisko
    ile_razy[czasopismo] = ile_razy.get(czasopismo, 0) + 1
    
print(czasopisma_osoby)
print(ile_razy)

