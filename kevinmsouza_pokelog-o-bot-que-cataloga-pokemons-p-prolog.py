for i in range(1):

    print('Nome tipo1 tipo2 evolução (OBS: se for só um tipo, insira os 2 iguais)')

    e=input().split(" ")

    nome=e[0]

    tipo1=e[1]

    tipo2=e[2]

    evolução=e[3]

    tipos= ['aço', 'água', 'dragão', 'elétrico','fada','fantasma','fogo','gelo','inseto', 'lutador', 'normal','pedra','planta','psiquico','sombrio','terrestre','venenoso','voador']

    

    print()

    print('pokemon('+nome+').')

    print('tipo('+nome+','+tipos[int(tipo1)-1]+').')

    if (tipo1!=tipo2):    

        print('tipo('+nome+','+tipos[int(tipo2)-1]+').')

    print()

    print('evolução('+nome+','+evolução+').')

    

    a=ag=dr=el=fa=fn=fg=gl=ins=lt=no=pe=pl=ps=es=te=ve=vo=0

    

    if (int(tipo1)==1):

        no+=1

        lt+=-1

        vo+=1

        ve+=2

        te+=-1

        pe+=1

        ins+=1

        fn+=0

        a+=1

        fg+=-1

        ag+=0

        pl+=1

        el+=0

        ps+=1

        gl+=1

        dr+=1

        es+=0

        fa+=1

    if (int(tipo1)==2):

        no+=0

        lt+=0

        vo+=0

        ve+=0

        te+=0

        pe+=0

        ins+=0

        fn+=0

        a+=1

        fg+=1

        ag+=1

        pl+=-1

        el+=-1

        ps+=0

        gl+=1

        dr+=0

        es+=0

        fa+=0

    if (int(tipo1)==3):

        no+=0

        lt+=0

        vo+=0

        ve+=0

        te+=0

        pe+=0

        ins+=0

        fn+=0

        a+=0

        fg+=1

        ag+=1

        pl+=1

        el+=1

        ps+=0

        gl+=-1

        dr+=-1

        es+=0

        fa+=-1

    if (int(tipo1)==4):

        no+=0

        lt+=0

        vo+=1

        ve+=0

        te+=-1

        pe+=0

        ins+=0

        fn+=0

        a+=1

        fg+=0

        ag+=0

        pl+=0

        el+=1

        ps+=0

        gl+=0

        dr+=0

        es+=0

        fa+=0

    if (int(tipo1)==5):

        no+=0

        lt+=1

        vo+=0

        ve+=-1

        te+=0

        pe+=0

        ins+=1

        fn+=0

        a+=-1

        fg+=0

        ag+=0

        pl+=0

        el+=0

        ps+=0

        gl+=0

        dr+=2

        es+=1

        fa+=0

    if (int(tipo1)==6):

        no+=2

        lt+=2

        vo+=0

        ve+=1

        te+=0

        pe+=0

        ins+=1

        fn+=-1

        a+=0

        fg+=0

        ag+=0

        pl+=0

        el+=0

        ps+=0

        gl+=0

        dr+=0

        es+=-1

        fa+=0

    if (int(tipo1)==7):

        no+=0

        lt+=0

        vo+=0

        ve+=0

        te+=-1

        pe+=-1

        ins+=1

        fn+=0

        a+=1

        fg+=1

        ag+=-1

        pl+=1

        el+=0

        ps+=0

        gl+=1

        dr+=0

        es+=0

        fa+=1

    if (int(tipo1)==8):

        no+=0

        lt+=-1

        vo+=0

        ve+=0

        te+=0

        pe+=-1

        ins+=0

        fn+=0

        a+=-1

        fg+=-1

        ag+=0

        pl+=0

        el+=0

        ps+=0

        gl+=1

        dr+=0

        es+=0

        fa+=0

    if (int(tipo1)==9):

        no+=0

        lt+=1

        vo+=-1

        ve+=0

        te+=1

        pe+=-1

        ins+=0

        fn+=0

        a+=0

        fg+=-1

        ag+=0

        pl+=1

        el+=0

        ps+=0

        gl+=0

        dr+=0

        es+=0

        fa+=0

    if (int(tipo1)==10):

        no+=0

        lt+=0

        vo+=-1

        ve+=0

        te+=0

        pe+=1

        ins+=1

        fn+=0

        a+=0

        fg+=0

        ag+=0

        pl+=0

        el+=0

        ps+=-1

        gl+=0

        dr+=0

        es+=1

        fa+=-1

    if (int(tipo1)==11):

        no+=1

        lt+=-1

        vo+=0

        ve+=0

        te+=0

        pe+=0

        ins+=0

        fn+=2

        a+=0

        fg+=0

        ag+=0

        pl+=0

        el+=0

        ps+=0

        gl+=0

        dr+=0

        es+=0

        fa+=0

    if (int(tipo1)==12):

        no+=1

        lt+=-1

        vo+=1

        ve+=1

        te+=-1

        pe+=0

        ins+=0

        fn+=0

        a+=-1

        fg+=1

        ag+=-1

        pl+=-1

        el+=0

        ps+=0

        gl+=0

        dr+=0

        es+=0

        fa+=0

    if (int(tipo1)==13):

        no+=0

        lt+=0

        vo+=-1

        ve+=-1

        te+=1

        pe+=0

        ins+=-1

        fn+=0

        a+=0

        fg+=-1

        ag+=1

        pl+=1

        el+=1

        ps+=0

        gl+=-1

        dr+=0

        es+=0

        fa+=0

    if (int(tipo1)==14):

        no+=0

        lt+=1

        vo+=0

        ve+=0

        te+=0

        pe+=0

        ins+=-1

        fn+=-1

        a+=0

        fg+=0

        ag+=0

        pl+=0

        el+=0

        ps+=1

        gl+=0

        dr+=0

        es+=-1

        fa+=0

    if (int(tipo1)==15):

        no+=0

        lt+=-1

        vo+=0

        ve+=0

        te+=0

        pe+=0

        ins+=-1

        fn+=1

        a+=0

        fg+=0

        ag+=0

        pl+=0

        el+=0

        ps+=2

        gl+=0

        dr+=0

        es+=1

        fa+=-1

    if (int(tipo1)==16):

        no+=0

        lt+=0

        vo+=0

        ve+=1

        te+=0

        pe+=1

        ins+=0

        fn+=0

        a+=0

        fg+=0

        ag+=-1

        pl+=-1

        el+=2

        ps+=0

        gl+=-1

        dr+=0

        es+=0

        fa+=0

    if (int(tipo1)==17):

        no+=0

        lt+=1

        vo+=0

        ve+=1

        te+=-1

        pe+=0

        ins+=1

        fn+=0

        a+=0

        fg+=0

        ag+=0

        pl+=1

        el+=0

        ps+=-1

        gl+=0

        dr+=0

        es+=0

        fa+=1

    if (int(tipo1)==18):

        no+=0

        lt+=1

        vo+=0

        ve+=0

        te+=2

        pe+=-1

        ins+=1

        fn+=0

        a+=0

        fg+=0

        ag+=0

        pl+=1

        el+=-1

        ps+=0

        gl+=-1

        dr+=0

        es+=0

        fa+=0

    if (int(tipo2)==1):

        no+=1

        lt+=-1

        vo+=1

        ve+=2

        te+=-1

        pe+=1

        ins+=1

        fn+=0

        a+=1

        fg+=-1

        ag+=0

        pl+=1

        el+=0

        ps+=1

        gl+=1

        dr+=1

        es+=0

        fa+=1

    if (int(tipo2)==2):

        no+=0

        lt+=0

        vo+=0

        ve+=0

        te+=0

        pe+=0

        ins+=0

        fn+=0

        a+=1

        fg+=1

        ag+=1

        pl+=-1

        el+=-1

        ps+=0

        gl+=1

        dr+=0

        es+=0

        fa+=0

    if (int(tipo2)==3):

        no+=0

        lt+=0

        vo+=0

        ve+=0

        te+=0

        pe+=0

        ins+=0

        fn+=0

        a+=0

        fg+=1

        ag+=1

        pl+=1

        el+=1

        ps+=0

        gl+=-1

        dr+=-1

        es+=0

        fa+=-1

    if (int(tipo2)==4):

        no+=0

        lt+=0

        vo+=1

        ve+=0

        te+=-1

        pe+=0

        ins+=0

        fn+=0

        a+=1

        fg+=0

        ag+=0

        pl+=0

        el+=1

        ps+=0

        gl+=0

        dr+=0

        es+=0

        fa+=0

    if (int(tipo2)==5):

        no+=0

        lt+=1

        vo+=0

        ve+=-1

        te+=0

        pe+=0

        ins+=1

        fn+=0

        a+=-1

        fg+=0

        ag+=0

        pl+=0

        el+=0

        ps+=0

        gl+=0

        dr+=2

        es+=1

        fa+=0

    if (int(tipo2)==6):

        no+=2

        lt+=2

        vo+=0

        ve+=1

        te+=0

        pe+=0

        ins+=1

        fn+=-1

        a+=0

        fg+=0

        ag+=0

        pl+=0

        el+=0

        ps+=0

        gl+=0

        dr+=0

        es+=-1

        fa+=0

    if (int(tipo2)==7):

        no+=0

        lt+=0

        vo+=0

        ve+=0

        te+=-1

        pe+=-1

        ins+=1

        fn+=0

        a+=1

        fg+=1

        ag+=-1

        pl+=1

        el+=0

        ps+=0

        gl+=1

        dr+=0

        es+=0

        fa+=1

    if (int(tipo2)==8):

        no+=0

        lt+=-1

        vo+=0

        ve+=0

        te+=0

        pe+=-1

        ins+=0

        fn+=0

        a+=-1

        fg+=-1

        ag+=0

        pl+=0

        el+=0

        ps+=0

        gl+=1

        dr+=0

        es+=0

        fa+=0

    if (int(tipo2)==9):

        no+=0

        lt+=1

        vo+=-1

        ve+=0

        te+=1

        pe+=-1

        ins+=0

        fn+=0

        a+=0

        fg+=-1

        ag+=0

        pl+=1

        el+=0

        ps+=0

        gl+=0

        dr+=0

        es+=0

        fa+=0

    if (int(tipo2)==10):

        no+=0

        lt+=0

        vo+=-1

        ve+=0

        te+=0

        pe+=1

        ins+=1

        fn+=0

        a+=0

        fg+=0

        ag+=0

        pl+=0

        el+=0

        ps+=-1

        gl+=0

        dr+=0

        es+=1

        fa+=-1

    if (int(tipo2)==11):

        no+=1

        lt+=-1

        vo+=0

        ve+=0

        te+=0

        pe+=0

        ins+=0

        fn+=2

        a+=0

        fg+=0

        ag+=0

        pl+=0

        el+=0

        ps+=0

        gl+=0

        dr+=0

        es+=0

        fa+=0

    if (int(tipo2)==12):

        no+=1

        lt+=-1

        vo+=1

        ve+=1

        te+=-1

        pe+=0

        ins+=0

        fn+=0

        a+=-1

        fg+=1

        ag+=-1

        pl+=-1

        el+=0

        ps+=0

        gl+=0

        dr+=0

        es+=0

        fa+=0

    if (int(tipo2)==13):

        no+=0

        lt+=0

        vo+=-1

        ve+=-1

        te+=1

        pe+=0

        ins+=-1

        fn+=0

        a+=0

        fg+=-1

        ag+=1

        pl+=1

        el+=1

        ps+=0

        gl+=-1

        dr+=0

        es+=0

        fa+=0

    if (int(tipo2)==14):

        no+=0

        lt+=1

        vo+=0

        ve+=0

        te+=0

        pe+=0

        ins+=-1

        fn+=-1

        a+=0

        fg+=0

        ag+=0

        pl+=0

        el+=0

        ps+=1

        gl+=0

        dr+=0

        es+=-1

        fa+=0

    if (int(tipo2)==15):

        no+=0

        lt+=-1

        vo+=0

        ve+=0

        te+=0

        pe+=0

        ins+=-1

        fn+=1

        a+=0

        fg+=0

        ag+=0

        pl+=0

        el+=0

        ps+=2

        gl+=0

        dr+=0

        es+=1

        fa+=-1

    if (int(tipo2)==16):

        no+=0

        lt+=0

        vo+=0

        ve+=1

        te+=0

        pe+=1

        ins+=0

        fn+=0

        a+=0

        fg+=0

        ag+=-1

        pl+=-1

        el+=2

        ps+=0

        gl+=-1

        dr+=0

        es+=0

        fa+=0

    if (int(tipo2)==17):

        no+=0

        lt+=1

        vo+=0

        ve+=1

        te+=-1

        pe+=0

        ins+=1

        fn+=0

        a+=0

        fg+=0

        ag+=0

        pl+=1

        el+=0

        ps+=-1

        gl+=0

        dr+=0

        es+=0

        fa+=1

    if (int(tipo2)==18):

        no+=0

        lt+=1

        vo+=0

        ve+=0

        te+=2

        pe+=-1

        ins+=1

        fn+=0

        a+=0

        fg+=0

        ag+=0

        pl+=1

        el+=-1

        ps+=0

        gl+=-1

        dr+=0

        es+=0

        fa+=0

    if (no<0):

        print('fraqueza('+nome+',normal).')

    if (lt<0):

        print('fraqueza('+nome+',lutador).')

    if (vo<0):

        print('fraqueza('+nome+',voador).')

    if (ve<0):

        print('fraqueza('+nome+',venenoso).')

    if (te<0):

        print('fraqueza('+nome+',terrestre).')

    if (pe<0):

        print('fraqueza('+nome+',pedra).')

    if (ins<0):

        print('fraqueza('+nome+',inseto).')

    if (fn<0):

        print('fraqueza('+nome+',fantasma).')

    if (a<0):

        print('fraqueza('+nome+',aço).')

    if (fg<0):

        print('fraqueza('+nome+',fogo).')

    if (ag<0):

        print('fraqueza('+nome+',água).')

    if (pl<0):

        print('fraqueza('+nome+',planta).')

    if (el<0):

        print('fraqueza('+nome+',elétrico).')

    if (ps<0):

        print('fraqueza('+nome+',psiquico).')

    if (gl<0):

        print('fraqueza('+nome+',gelo).')

    if (dr<0):

        print('fraqueza('+nome+',dragão).')

    if (es<0):

        print('fraqueza('+nome+',sombrio).')

    if (fa<0):

        print('fraqueza('+nome+',fada).')