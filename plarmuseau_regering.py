##### import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



partij=pd.DataFrame([

    ['nva',25,1,4],

    ['ps',20,2,-3],

    ['vb',18,1,9],

    ['mr',14,2,1],

    ['ecolo',13,2,-4],

    ['cdv',12,1,0],

    ['ovld',12,1,1],

    ['spa',9,1,-3],

    ['groen',8,1,-4],

    ['cdh',5,2,0],

    ['defi',2,2,4],

    ['pvda',3,1,-6],

    ['ptb',9,2,-6]

],columns=['partij','zetels','vlw','spectr'])
partij
partij.groupby('vlw').sum(),partij.mean()
partij.min()
def checkevenwicht(partijsel,verbose):

    partijsel=partijsel.sort_values('zetels',ascending=False)

    meerderheid=partijsel.sum().zetels>76

    #partijsel['weight']=partijsel['zetels']*partijsel['spectr']

    leftmax=partijsel.min().spectr

    rightmax=partijsel.max().spectr

    partijgap=(rightmax-leftmax)<13

    vlwverd=partijsel.groupby('vlw').sum()

    if len(vlwverd)==2:

        meerderheidvl=vlwverd.iat[0,0]>44

        meerderheidw=vlwverd.iat[1,0]>33

        vlwgap=vlwverd.iat[0,1]-vlwverd.iat[1,1]

    else:

        meerderheidvl=False

        meerderheidw=False

        vlwgap=False

    samenstelling=partijsel.sum().partij

    vlwgap=vlwgap<25



    if verbose:

        print('meerderheid be',meerderheid)

        print('meerderheid vl',meerderheidvl )

        print('meerderheid w', meerderheidw)

        print('partijkloof klein',partijgap,leftmax,rightmax,rightmax-leftmax)

        print('vl -w kloof',vlwgap)

        print('',vlwverd)

    return meerderheid&meerderheidvl&meerderheidw&partijgap&vlwgap,samenstelling

checkevenwicht(partij.iloc[[0,1,2,3,4,5]],True)
selectie=list(partij.index.values)

regeringsel=[]



mogelijkheden=[]



for ai in selectie:

    regeringsel.append(ai)

    selectieb=selectie

    selectieb.remove(ai)



    for bi in selectieb:

        regeringsel.append(bi)

        selectiec=selectieb

        selectiec.remove(bi)



        for ci in selectiec:

            regeringsel.append(ci)

            selectied=selectiec

            selectied.remove(ci)



            for di in selectied:

                regeringsel.append(di)

                selectiee=selectied

                selectiee.remove(di)



                for ei in selectiee:

                    regeringsel.append(ei)

                    selectief=selectiee

                    selectief.remove(ei)

                    

                    

                    for fi in selectief:

                        regeringsel.append(fi)

                        #selectieg=selectief

                        #selectieg.remove(fi)

                        

                        #for gi in selectieg:

                        #    regeringsel.append(gi)

                            #selectie.remove(fi)

                        if True:

                            evenw,naam=checkevenwicht(partij.iloc[regeringsel],False)

                            if evenw:

                                print('we got a winner:',naam)

                                mogelijkheden.append(naam)

                            #selectie.append(fi)

                        #    regeringsel.remove(gi)

                        #selectie.append(fi)

                        regeringsel.remove(fi)

                    

                    #selectief=selectiee

                    regeringsel.remove(ei)

                    

                #selectiee=selectied

                regeringsel.remove(di)



            

            #selectied=selectiec

            regeringsel.remove(ci)



        #selectiec=selectieb

        regeringsel.remove(bi)

        

    #selectieb=selectie

    regeringsel.remove(ai)

    