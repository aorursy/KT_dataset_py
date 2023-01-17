from mpl_toolkits.mplot3d import Axes3D
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt # plotting
import numpy as np # linear algebra
import os # accessing directory structure
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
tickers = {
    "ABAD1":"59612098290740355",
    "ABDI1":"49054891736433700",
    "ALBZ1":"57944184894703821",
    "ALIR1":"65004959184388996",
    "ALVN1":"32678431934327184",
    "AMIN1":"50100062518826135",
    "AMLH1":"16959429956899455",
    "APPE1":"55254206302462116",
    "APOZ1":"55127657985997520", 
    "ARDK1":"4614779520007780",
    "ASAL1":"63363116407864462",
    "ASIA1":"51106317433079213",
    "ATIR1":"7483280423474368",
    "AYEG1":"15039949673085566",
    "AZAB1":"38547060135156069",
    "AZIN1":"42075223783409640",
    "AZIX1":"22129017544200",
    "BAHN1":"66772024744156373",
    "BALB1":"70270965300262393",
    "BALI1":"28328710198554144",
    "BAMA1":"4942127026063388",
    "BANK1":"48010225447410247",
    "BARZ1":"23214828924506640",
    "BDAN1":"48511238766369097",
    "BENN1":"17059960254855208",
    "BFJR1":"46178280540110577",
    "BHMN1":"26824673819862694",
    "BIME1":"11773403764702778",
    "BIMX1":"59735609117437896",
    "BKHZ1":"47333458678352378",
    "BMEL1":"13611044044646901",
    "BMLT1":"778253364357513#",
    "BMPS1":"61102694810476197",
    "BOTA1":"30650426998863332",
    "BPAR1":"33293588228706998",
    "BPAS1":"9536587154100457#",
    "BPST1":"22087269603540841",
    "BRKT1":"34557241988629814",
    "BROJ1":"60061422939859083",
    "BSDR1":"28320293733348826",
    "BSTE1":"17800036702302776",
    "BTEJ1":"63917421733088077",
    "BVMA1":"29860265627578401",
    "CHAR1":"33783140337377394",
    "CHCH1":"44850033148208596",
    "CHDN1":"12329519546621752",
    "CHML1":"18027801615184692",
    "CIDC1":"37281199178613855",
    "COMB1":"30719054967088301",
    "DABO1":"61332057061846617",
    "DADE1":"65999092673039059",
    "DALZ1":"60451823714332895",
    "DAML1":"66450490505950110",
    "DARO1":"7183333492448248#",
    "DDPK1":"33603212156438463",
    "DFRB1":"56550776668133562",
    "DJBR1":"33406621820337161",
    "DKSR1":"23353689102956991",
    "DLGM1":"29247915161590165",
    "DMOR1":"2434703913394836#",
    "DMVN1":"3623921205367364#",
    "DODE1":"40611478183231802",
    "DOSE1":"12387472624849835",
    "DPAK1":"67988012428906654",
    "DRKH1":"24079409192818584",
    "DRZK1":"22255783119783047",
    "DSBH1":"5866848234665627#",
    "DSIN1":"11432067920374603",
    "DSOB1":"43622578471330344",
    "DTIP1":"29758477602878557",
    "DZAH1":"8915450910866216#",
    "EPRS1":"41048299027409941",
    "EXIR1":"4384288570322406#",
    "FAIR1":"40808043719554948",
    "FAIX1":"53249498382937694",
    "FAJR1":"41302553376174581",
    "FKAS1":"4733285133017464#",
    "FKHZ1":"28864540805361867",
    "FNAR1":"32821908911812078",
    "FOLD1":"46348559193224090",
    "FRIS1":"54419429862704331",
    "FRVR1":"408934423224097#",
    "FTIR1":"18303237082155264",
    "GBEH1":"46982154647719707",
    "GBJN1":"71523986304961239",
    "GCOZ1":"22299894048845903",
    "GDIR1":"26014913469567886",
    "GESF1":"40411537531154482",
    "GGAZ1":"15259343650667588",
    "GHAT1":"70289374539527245",
    "GHEG1":"14398278072324784",
    "GHND1":"37631109616997982",
    "GLOR1":"56820995669577571",
    "GMEL1":"41796741644273824",
    "GMRO1":"43342306308122676",
    "GNBO1":"63380098535169030",
    "GOLG1":"35700344742885862",
    "GORJ1":"31024260997481994",
    "GORX1":"53482856220074779",
    "GOST1":"48990026850202503",
    "GPSH1":"67030488744129337",
    "HFRS1":"35424116338766901",
    "HJPT1":"16369313804633525",
    "HMRZ1":"68635710163497089",
    "HSHM1":"28809886765682162",
    "HTOK1":"22260326095996531",
    "HWEB1":"43362635835198978",
    "IAGM1":"23838634016123354",
    "IDOC1":"47841327496247362",
    "IKCO1":"65883838195688438",
    "IKHR1":"7395271748414592#",
    "INFO1":"40505767672724777",
    "IPAR1":"9481703061634967#",
    "IPTR1":"69143674941561637",
    "IRDR1":"69090868458637360",
    "JAMD1":"30765727085936322",
    "JHRM1":"33629260529503413",
    "JOSH1":"70219663893822560",
    "JPPC1":"27096851668435724",
    "KALA1":"44549439964296944",
    "KALZ1":"62952165421099192",
    "KCHI1":"16405556680571453",
    "KDPS1":"2254054929817435#",
    "KFAN1":"28033133021443774",
    "KGND1":"22382156782768756",
    "KHOC1":"41974758296041288",
    "KHSH1":"63915926161403347",
    "KIMI1":"20024911381434086",
    "KLBR1":"24303422207378456",
    "KNRZ1":"20411759370751096",
    "KRAF1":"47996917271187218",
    "KRIR1":"59217041815333317",
    "KRSN1":"43552974795606067",
    "KRTI1":"53113471126689455",
    "KSIM1":"66701874099226162",
    "KSKA1":"24254843881948059",
    "KVEH1":"60350996279289099",
    "KVEX1":"59525964192678026",
    "KVRZ1":"7235435095059069#",
    "LAMI1":"25286509736208688",
    "LAPS1":"69454539056549106",
    "LEAB1":"39116664428676213",
    "LENT1":"14957056743925737",
    "LIRZ1":"3149396562827132#",
    "LKGH1":"23086515493897579",
    "LMIR1":"48623320733330408",
    "LPAK1":"34032872653290886",
    "LSMD1":"71744682148776880",
    "LTOS1":"59142194115401696",
    "LTOX1":"43455612336353491",
    "LZIN1":"20946530370469828",
    "MADN1":"58931793851445922",
    "MAGS1":"5054819322815158#",
    "MAPN1":"67126881188552864",
    "MARK1":"20865316761157979",
    "MESI1":"31879190587976736",
    "MHKM1":"17330546482145553",
    "MNGZ1":"50341528161302545",
    "MNMH1":"15826229869421585",
    "MNSR1":"17834623106317041",
    "MOBN1":"27922860956133067",
    "MOTJ1":"22086876724551482",
    "MRAM1":"6131290133202745#",
    "MRGN1":"52975109254504632",
    "MRIN1":"30231789123900526",
    "MSKN1":"3863538898378476#",
    "MSMI1":"35425587644337450",
    "MSTI1":"57273529732791251",
    "NAFT1":"33931218652865616",
    "NALM1":"57875847776839336",
    "NASI1":"23837844039713715",
    "NBEH1":"22667016906590506",
    "NGFO1":"56324206651661881",
    "NIKI1":"25336820825905643",
    "NIRO1":"20453828618330936",
    "NKOL1":"62177651435283872",
    "NMOH1":"39436183727126211",
    "NOVN1":"47302318535715632",
    "NPRS1":"14073782708315535",
    "NSAZ1":"49353447565507376",
    "NSTH1":"32845891587040106",
    "OFOG1":"49502666250908008",
    "OFRS1":"30852391633490755",
    "OFST1":"23936607891892333",
    "OIMC1":"52232388263291380",
    "OMID1":"18599703143458101",
    "PABX1":"65472108074101196",
    "PAKS1":"11622051128546106",
    "PARK1":"7711282667602555#",
    "PARS1":"6110133418282108#",
    "PASH1":"62786156501584862",
    "PASN1":"23441366113375722",
    "PDRO1":"70474983732269112",
    "PELC1":"5187018329202415#",
    "PETR1":"59486059679335017",
    "PFAN1":"65122215875355555",
    "PIAZ1":"35158826900216508",
    "PIRN1":"43062880954780884",
    "PJMZ1":"32357363984168442",
    "PKER1":"38437201078089290",
    "PKHA1":"70934270174405743",
    "PKLJ1":"25244329144808274",
    "PKOD1":"42354736493447489",
    "PLKK1":"57722642338781674",
    "PMSZ1":"6043384171800349#",
    "PNBA1":"35366681030756042",
    "PNES1":"7745894403636165#",
    "PNTB1":"48753732042176709",
    "PRDZ1":"20562694899904339",
    "PRKT1":"59607545337891226",
    "PSER1":"20560887114747719",
    "PSHZ1":"38568786927478796",
    "PSIR1":"57639364758870873",
    "PTAP1":"22560050433388046",
    "PTEH1":"51617145873056483",
    "RENA1":"7385624172574740#",
    "RIIR1":"12752224677923341",
    "RIIX1":"12041217376571987",
    "RINM1":"45895339414786358",
    "RKSH1":"27952969918967492",
    "ROOI1":"22787503301679573",
    "ROZD1":"40262275031537922",
    "RSAP1":"45174198424472334",
    "RTIR1":"22903901709044823",
    "SADB1":"34890845654517313",
    "SADR1":"23293437377896568",
    "SAHD1":"63481599728522324",
    "SAKH1":"25514780181345713",
    "SAMA1":"34673681828119297",
    "SAND1":"37204371816016200",
    "SBAH1":"18063426072758458",
    "SBEH1":"42387718866026650",
    "SBHN1":"32525655729432562",
    "SBOJ1":"66295665969375744",
    "SDAB1":"4563413583000719#",
    "SDOR1":"27218386411183410",
    "SDST1":"27000326841257664",
    "SEPA1":"8977441217024425#",
    "SEPK1":"71856634742001725",
    "SEPP1":"49188729526980541",
    "SFKZ1":"15521712617204216",
    "SFNO1":"4528607775462304#",
    "SFRS1":"41227201752535311",
    "SGAZ1":"62346804681275278",
    "SGEN1":"60654872678917533",
    "SGOS1":"66456062140680461",
    "SGRB1":"52220424531578944",
    "SHAD1":"20487994977117557",
    "SHFA1":"36899214178084525",
    "SHFS1":"43781018754867729",
    "SHGN1":"41284516796232939",
    "SHKR1":"35964395659427029",
    "SHMD1":"67206358287598044",
    "SHND1":"10120557300120078",
    "SHOY1":"3493306453706327#",
    "SHPZ1":"59921975187856916",
    "SHSI1":"30974710508383145",
    "SHZG1":"29747059672582491",
    "SIMS1":"6757220448540984#",
    "SIMX1":"23458117616920867",
    "SINA1":"25001509088465005",
    "SKAZ1":"67327029014085707",
    "SKBV1":"11258722998911897",
    "SKER1":"15472396110662150",
    "SKHS1":"4470657233334072#",
    "SKOR1":"65321970913593427",
    "SLMN1":"28672095850798501",
    "SMAZ1":"33808206014018431",
    "SMRG1":"28450080638096732",
    "SNMA1":"57309221039930244",
    "SNRO1":"62603302940123327",
    "SORB1":"54277068923045214",
    "SPAH1":"2328862017676109#",
    "SPKH1":"31791737198597563",
    "SPPE1":"61506294208022391",
    "SPTA1":"68488673556087148",
    "SROD1":"11964419322927535",
    "SSAP1":"37614886280396031",
    "SSHR1":"26997316501080743",
    "SSIN1":"35796086458096255",
    "SSNR1":"14231831499205396",
    "SSOF1":"13227300125161435",
    "STEH1":"30829203706095076",
    "SURO1":"15949743338644220",
    "SWIC1":"47377315952751604",
    "SYSM1":"47749661205825616",
    "SZPO1":"4758266259250794#",
    "TAIR1":"41935584690956944",
    "TAMI1":"67690708346979840",
    "TAYD1":"3722699128879020#",
    "TAZB1":"1358190916156744#",
    "TBAS1":"8977369674477111#",
    "TGOS1":"68117765376081366",
    "TKIN1":"3823243780502959#",
    "TKNO1":"3654864906585643#",
    "TKSM1":"24085906177899789",
    "TMEL1":"17528249960294496",
    "TMKH1":"7457232989848872#",
    "TMVD1":"34641719089573667",
    "TNOV1":"25357135030606405",
    "TOKA1":"47232550823972469",
    "TOSA1":"2944500421562364#",
    "TRIR1":"19298748452450329",
    "TRIX1":"69858202392669780",
    "TRNS1":"46752599569017089",
    "TSBE1":"13937270451301973",
    "TSHE1":"54676885047867737",
    "TSRZ1":"57086055330734195",
    "VLMT1":"11403770140000603",
    "VSIN1":"45050389997905274",
    "YASA1":"63580313877463104",
    "YASX1":"32338211917133256",
    "ZMYD1":"2589887561569709#",
    "ZPRS1":"33420285433308219"
    }
def stock(ticker=""):
    urlid = tickers[ticker]
    url = 'http://www.tsetmc.com/tsev2/data/Export-txt.aspx?t=i&a=1&b=0&i=' + urlid 
    df = pd.read_csv(url)
    df=df.loc[::-1].reset_index(drop=True)
    return (df)
# To get live data inside Iran (Interanet only, international IPs are blocked.)

for ticker in tickers:
    globals()['Stock_%s' % ticker] = stock(ticker)
    globals()['Stock_%s' % ticker].to_csv('Stock_%s' % ticker+'.csv')
Stock_ZPRS1 = pd.read_csv('/kaggle/input/Stock_ZPRS1.csv')
plt.plot(Stock_ZPRS1['<CLOSE>'] ,label='ZPRS1')
plt.title("Ticker Close Prices")
plt.legend()
plt.show()
# Distribution graphs (histogram/bar graph) of column data
def plotPerColumnDistribution(df, nGraphShown, nGraphPerRow):
    nunique = df.nunique()
    df = df[[col for col in df if nunique[col] > 1 and nunique[col] < 50]] # For displaying purposes, pick columns that have between 1 and 50 unique values
    nRow, nCol = df.shape
    columnNames = list(df)
    nGraphRow = (nCol + nGraphPerRow - 1) / nGraphPerRow
    plt.figure(num = None, figsize = (6 * nGraphPerRow, 8 * nGraphRow), dpi = 80, facecolor = 'w', edgecolor = 'k')
    for i in range(min(nCol, nGraphShown)):
        plt.subplot(nGraphRow, nGraphPerRow, i + 1)
        columnDf = df.iloc[:, i]
        if (not np.issubdtype(type(columnDf.iloc[0]), np.number)):
            valueCounts = columnDf.value_counts()
            valueCounts.plot.bar()
        else:
            columnDf.hist()
        plt.ylabel('counts')
        plt.xticks(rotation = 90)
        plt.title(f'{columnNames[i]} (column {i})')
    plt.tight_layout(pad = 1.0, w_pad = 1.0, h_pad = 1.0)
    plt.show()

# Correlation matrix
def plotCorrelationMatrix(df, graphWidth):
    filename = df.dataframeName
    df = df.dropna('columns') # drop columns with NaN
    df = df[[col for col in df if df[col].nunique() > 1]] # keep columns where there are more than 1 unique values
    if df.shape[1] < 2:
        print(f'No correlation plots shown: The number of non-NaN or constant columns ({df.shape[1]}) is less than 2')
        return
    corr = df.corr()
    plt.figure(num=None, figsize=(graphWidth, graphWidth), dpi=80, facecolor='w', edgecolor='k')
    corrMat = plt.matshow(corr, fignum = 1)
    plt.xticks(range(len(corr.columns)), corr.columns, rotation=90)
    plt.yticks(range(len(corr.columns)), corr.columns)
    plt.gca().xaxis.tick_bottom()
    plt.colorbar(corrMat)
    plt.title(f'Correlation Matrix for {filename}', fontsize=15)
    plt.show()

# Scatter and density plots
def plotScatterMatrix(df, plotSize, textSize):
    df = df.select_dtypes(include =[np.number]) # keep only numerical columns
    # Remove rows and columns that would lead to df being singular
    df = df.dropna('columns')
    df = df[[col for col in df if df[col].nunique() > 1]] # keep columns where there are more than 1 unique values
    columnNames = list(df)
    if len(columnNames) > 10: # reduce the number of columns for matrix inversion of kernel density plots
        columnNames = columnNames[:10]
    df = df[columnNames]
    ax = pd.plotting.scatter_matrix(df, alpha=0.75, figsize=[plotSize, plotSize], diagonal='kde')
    corrs = df.corr().values
    for i, j in zip(*plt.np.triu_indices_from(ax, k = 1)):
        ax[i, j].annotate('Corr. coef = %.3f' % corrs[i, j], (0.8, 0.2), xycoords='axes fraction', ha='center', va='center', size=textSize)
    plt.suptitle('Scatter and Density Plot')
    plt.show()

nRowsRead = 1000 # specify 'None' if want to read whole file
# Stock_ABAD1.csv may have more rows in reality, but we are only loading/previewing the first 1000 rows
df1 = pd.read_csv('/kaggle/input/Stock_ABAD1.csv', delimiter=',', nrows = nRowsRead)
df1.dataframeName = 'Stock_ABAD1.csv'
nRow, nCol = df1.shape
print(f'There are {nRow} rows and {nCol} columns')
df1.head(5)
plotPerColumnDistribution(df1, 10, 5)
plotCorrelationMatrix(df1, 8)
plotScatterMatrix(df1, 20, 10)
nRowsRead = 1000 # specify 'None' if want to read whole file
# Stock_ABDI1.csv may have more rows in reality, but we are only loading/previewing the first 1000 rows
df2 = pd.read_csv('/kaggle/input/Stock_ABDI1.csv', delimiter=',', nrows = nRowsRead)
df2.dataframeName = 'Stock_ABDI1.csv'
nRow, nCol = df2.shape
print(f'There are {nRow} rows and {nCol} columns')
df2.head(5)
plotPerColumnDistribution(df2, 10, 5)
plotCorrelationMatrix(df2, 8)
plotScatterMatrix(df2, 20, 10)
nRowsRead = 1000 # specify 'None' if want to read whole file
# Stock_ALBZ1.csv may have more rows in reality, but we are only loading/previewing the first 1000 rows
df3 = pd.read_csv('/kaggle/input/Stock_ALBZ1.csv', delimiter=',', nrows = nRowsRead)
df3.dataframeName = 'Stock_ALBZ1.csv'
nRow, nCol = df3.shape
print(f'There are {nRow} rows and {nCol} columns')
df3.head(5)
plotPerColumnDistribution(df3, 10, 5)
plotCorrelationMatrix(df3, 8)
plotScatterMatrix(df3, 20, 10)