from IPython.display import IFrame
IFrame('https://app.powerbi.com/view?r=eyJrIjoiYWUyZjJhZjQtMDIwNi00OWQ0LTliNDctZmNiM2Q5YTkzNzJhIiwidCI6ImRjMWYwNGY1LWMxZTUtNDQyOS1hODEyLTU3OTNiZTQ1YmY5ZCIsImMiOjEwfQ%3D%3D', width=800, height=500)
UseSciSpacy=True
import numpy as np
import pylab
import pandas as pd
import json
import os
import re
import spacy
import numpy as np  
import pandas as pd 
import spacy

if(UseSciSpacy):
   #Instal SciSpacy
    !pip install -U scispacy
    !pip install -U https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.2.4/en_core_sci_lg-0.2.4.tar.gz
    import scispacy
    nlp=spacy.load("/opt/conda/lib/python3.6/site-packages/en_core_sci_lg/en_core_sci_lg-0.2.3/", disable=["tagger"])
else:
    !python -m spacy download en_core_web_lg
    nlp = spacy.load('en_core_web_lg')
# This flags determines whether to re-run the text matching code.
# Warning, if it takes along time! Intermediate data 
# from this step is provided with the notebook.
RunMatching=False
# Open the RxNorm file and extract drug names list
file=open("/kaggle/input/rxnorm-inputdata/RxNorm_full_prescribe_03022020/rrf/RXNCONSO.RRF",'r').readlines()
names=[]
for line in file:
    names.append(line.split("|")[14].lower())
names=np.unique(names)

# restrict to single-word drug names with >5 characters
singlenames=[]
for name in names:
    if ((not " " in name) and (len(name)>5)):
        singlenames.append(name)

# Load up elements
Elements=pd.read_csv("/kaggle/input/rxnorm-inputdata/Elements.csv")
ElementNames = Elements.Element.str.lower()

# Load up animals, 
AnimalsRaw=open("/kaggle/input/rxnorm-inputdata/animals.txt",'r').readlines()
AnimalNames=[]
for a in AnimalsRaw:
    if not " " in a:
        AnimalNames.append(a[:-1].lower())

# Load up fruit and veg
FruitVegRaw=open("/kaggle/input/rxnorm-inputdata/FruitAndVeg.txt",'r').readlines()
FruitVegNames=[]
for a in FruitVegRaw:
    if not " " in a:
        FruitVegNames.append(nlp(a[:-1].lower())[0].lemma_)

# Apply the filter        
filterednames=[]
for name in singlenames:
    if (not name  in AnimalNames) and (not name  in FruitVegNames)  and (not name  in ElementNames.values):
        filterednames.append(name)
        
np.savetxt("/kaggle/working/DrugNames.txt",filterednames,fmt="%s")
# These are helper functions for extracting word matches from the text
# both lemmatized and non-lemmatized versions are possible.

Paths=["/kaggle/input/CORD-19-research-challenge/noncomm_use_subset/noncomm_use_subset/pdf_json/","/kaggle/input/CORD-19-research-challenge/comm_use_subset/comm_use_subset/pdf_json/","/kaggle/input/CORD-19-research-challenge/biorxiv_medrxiv/biorxiv_medrxiv/pdf_json/","/kaggle/input/CORD-19-research-challenge/custom_license/custom_license/pdf_json/"]


# These functions determine what blocks are pulled from the paper for matching
def TitleBlocks(paper):
    return([{'text':paper['metadata']['title']}])

def AbstractBlocks(paper):
    return(paper['abstract'])

def BodyBlocks(paper):
    return(paper['body_text'])



# This function finds matching lemmas and notes positions of
# occurence in the relevant json block. This function uses
# the lemmatized text.
def PullMentionsLemmatized(Paths, BlockSelector,SecName, Words):

    Positions=[]
    FoundWords=[]
    Section=[]
    BlockID=[]
    BlockText=[]
    PaperID=[]
    
    tokenized_words=[]
    for w in Words:
        tokenized_words.append(nlp(w.lower())[0].lemma_)
    for Path in Paths:
        print(Path)

        Files=os.listdir(Path)
        for p in Files:

            readfile=open(Path+p,'r')
            paper=json.load(readfile)
            Blocks=BlockSelector(paper)

            for b in range(0,len(Blocks)):
                text=nlp(Blocks[b]['text'].lower())

                for t in text:
                    for w in tokenized_words:
                        if(w == t.lemma_):
                            Section.append(SecName)
                            FoundWords.append(w)
                            Positions.append(t.idx)
                            BlockText.append(Blocks[b]['text'])
                            BlockID.append(b)
                            PaperID.append(p[:-5])
    return {'sha':PaperID,'blockid':BlockID,'word':FoundWords,'sec':Section,'pos':Positions,'block':BlockText}


# This function finds matching words and notes positions of
# occurence in the relevant json block. This function uses
# direct text matching (not lemmatized)
def PullMentionsDirect(Paths, BlockSelector,SecName, Words):
    Positions=[]
    FoundWords=[]
    Section=[]
    BlockID=[]
    BlockText=[]
    PaperID=[]
    for wi in range(0,len(Words)):
        Words[wi]=Words[wi].lower()
    for Path in Paths:
        print(Path)

        Files=os.listdir(Path)
        for p in Files:

            readfile=open(Path+p,'r')
            paper=json.load(readfile)
            Blocks=BlockSelector(paper)

            for b in range(0,len(Blocks)):
                text=Blocks[b]['text'].lower()
                for w in Words:
                    if(w in text):
                        pos=text.find(w)
                   
                        #check we're not in the middle of another word
                        if(text[pos-1]==" " and ( (pos+len(w))>=len(text) or not text[pos+len(w)].isalpha())):
                            Section.append(SecName)
                            FoundWords.append(w)
                            Positions.append(text.find(w))
                            BlockText.append(Blocks[b]['text'])
                            BlockID.append(b)
                            PaperID.append(p[:-5])
    return {'sha':PaperID,'blockid':BlockID,'word':FoundWords,'sec':Section,'pos':Positions,'block':BlockText}


# Run to get treatment words
def ExtractToCSV(Words,Filename,Lemmatized=True, RunTitle=True, RunAbstract=True, RunBody=False):

    if(Lemmatized):
        PullMentions = PullMentionsLemmatized
    else:
        PullMentions = PullMentionsDirect
    
    DataDicts=[]
    if(RunTitle): 
        DataDicts.append(PullMentions(Paths, TitleBlocks,    "title",    Words))
    if(RunAbstract):
        DataDicts.append(PullMentions(Paths, AbstractBlocks, "abstract", Words))
    if(RunBody):
        DataDicts.append(PullMentions(Paths, BodyBlocks,     "body",     Words))

    SummedDictionary=DataDicts[0]
    for k in DataDicts[0].keys():
        for d in DataDicts:
            SummedDictionary[k]=SummedDictionary[k]+d[k]

    dat=pd.DataFrame(SummedDictionary)
    dat.to_csv(Filename)


#Switch this off to run over only title and abstract -
#  go faster for debugging, but less complete info.
IncludeBodyText=True

# These lines of code will run the extraction

if(RunMatching):
    Words=["COVID-19", "Coronavirus", "Corona", "2019-nCoV", "SARS-CoV",]
    ExtractToCSV(Words, "/kaggle/working/TitleAbstractBodyMatches_virusnames.csv", Lemmatized=False,RunBody=IncludeBodyText)

    Words=np.loadtxt("DrugNames.txt",dtype='str')
    ExtractToCSV(Words, "/kaggle/working/TitleAbstractBodyMatches_drugs.csv", Lemmatized=False,RunBody=IncludeBodyText)

    Words=['treat','treatment' 'alleviate', 'manage', 'suppress','suppression', 'prescribe','therapy','cure','remedy', 'therapeutic','administer']
    ExtractToCSV(Words, "/kaggle/working/TitleAbstractBodyMatches_therapies.csv", Lemmatized=True,RunBody=IncludeBodyText)

    Words=["vitro", "vivo", "in-vitro", "in-vivo", "mouse","mice","clinial","human","computational","vertical","horizontal","theoretical","simulation"]
    ExtractToCSV(Words, "/kaggle/working/TitleAbstractBodyMatches_exptypes.csv", Lemmatized=True,RunBody=IncludeBodyText)

dat_therapies=pd.read_csv("/kaggle/input/textmatchesvt/TitleAbstractBodyMatches_therapies.csv")
dat_drugs= pd.read_csv("/kaggle/input/textmatchesvt/TitleAbstractBodyMatches_drugs.csv")
dat_viruses= pd.read_csv("/kaggle/input/textmatchesvt/TitleAbstractBodyMatches_virusnames.csv")
dat_exps= pd.read_csv("/kaggle/input/textmatchesvt/TitleAbstractBodyMatches_exptypes.csv")
# Drop unnecessary columns
dat_drugs=dat_drugs.drop('Unnamed: 0',axis=1).set_index('block')
dat_therapies=dat_therapies.drop('Unnamed: 0',axis=1).set_index('block')
dat_viruses=dat_viruses.drop('Unnamed: 0',axis=1).set_index('block')
dat_exps=dat_exps.drop('Unnamed: 0',axis=1).set_index('block')
# We'll use this function later to see if two words are in the same sentence
#  within the block

def SameSentenceCheck(block,pos1,pos2):
    if(pos1<pos2):
        Interstring=block[int(pos1):int(pos2)]
    else:
        Interstring=block[int(pos2):int(pos1)]
    SentenceEnders=[".",";","?","!"]
    for s in SentenceEnders:
        if s in Interstring:
            return 0
    return 1
# This function makes the 2D quilt plot for showing co-occurences at block
#   or sentence level of various classes of search terms
#
def Make2DPlot(dat_joined, factor1, factor2, single_sentence_plots=False):
    if(single_sentence_plots):
        grouped = dat_joined[dat_joined.same_sentence==True].groupby(['word_'+factor1,'word_'+factor2])
    else:
        grouped = dat_joined.groupby(['word_'+factor1,'word_'+factor2])

    Values    = grouped.count().values[:,0]

    Index=grouped.count().index
    Index1=[]
    Index2=[]
    for i in Index:
        Index1.append(i[0])
        Index2.append(i[1])

    Uniq1=np.unique(Index1)
    Uniq2=np.unique(Index2)

    for i in range(0,len(Index1)):
        Index1[i]=np.where(Index1[i]==Uniq1)[0][0]
        Index2[i]=np.where(Index2[i]==Uniq2)[0][0]

    pylab.figure(figsize=(5,5),dpi=200)
    hist=pylab.hist2d(Index1,Index2, (range(0,len(Uniq1)+1),range(0,len(Uniq2)+1)), weights=Values,cmap='Blues')
    pylab.xticks(np.arange(0,len(Uniq1))+0.5, Uniq1,rotation=90)
    pylab.yticks(np.arange(0,len(Uniq2))+0.5, Uniq2)
    pylab.clim(0,np.max(hist[0])*1.5)
    for i in range(0,len(Uniq1)):
        for j in range(0,len(Uniq2)):
            pylab.text(i+0.5,j+0.5,int(hist[0][i][j]),ha='center',va='center')

    pylab.colorbar()
    if(single_sentence_plots):
        pylab.title(factor1+" and " +factor2+" in One Sentence")
        pylab.tight_layout()
        pylab.savefig("Overlap"+factor1+"_Vs_"+factor2+"_2D_sentence.png",bbox_inches='tight',dpi=200)
    else:
        pylab.title(factor1+" and " +factor2+" in One Block")
        pylab.tight_layout()
        pylab.savefig("Overlap"+factor1+"_Vs_"+factor2+"_2D_block.png",bbox_inches='tight',dpi=200)
# Prune and join, and extract overlap counts
dat_joined_vt=dat_therapies.join(dat_viruses, rsuffix='_virus',lsuffix="_therapy")
dat_joined_vt=dat_joined_vt[dat_joined_vt.notna().word_therapy & dat_joined_vt.notna().word_virus]


#Make single sentence index
dat_joined_vt=dat_joined_vt.drop(["sha_therapy","blockid_therapy","sec_therapy"],axis=1).reset_index().rename(columns={"sha_virus":"sha","blockid_virus":"blockid","sec_virus":"sec"})
SingleSentence=[]
for i in dat_joined_vt.index:
    SingleSentence.append(SameSentenceCheck(dat_joined_vt.block[i],dat_joined_vt.pos_virus[i],dat_joined_vt.pos_therapy[i]))
dat_joined_vt.insert(len(dat_joined_vt.columns),'same_sentence',SingleSentence)
dat_joined_vt.to_csv("Overlaps_Virus_Therapy.csv")

Make2DPlot(dat_joined_vt,"virus","therapy")
Make2DPlot(dat_joined_vt,"virus","therapy",single_sentence_plots=True)
# Prune and join, and extract overlap counts
dat_joined_vd=dat_drugs.join(dat_viruses, rsuffix='_virus',lsuffix="_drug")
dat_joined_vd=dat_joined_vd[dat_joined_vd.notna().word_drug & dat_joined_vd.notna().word_virus]

dat_joined_vd=dat_joined_vd.drop(["sha_drug","blockid_drug","sec_drug"],axis=1).reset_index().rename(columns={"sha_virus":"sha","blockid_virus":"blockid","sec_virus":"sec"})
SingleSentence=[]
for i in dat_joined_vd.index:
    SingleSentence.append(SameSentenceCheck(dat_joined_vd.block[i],dat_joined_vd.pos_drug[i],dat_joined_vd.pos_drug[i]))
dat_joined_vd.insert(len(dat_joined_vd.columns),'same_sentence',SingleSentence)
dat_joined_vd.to_csv("Overlaps_Virus_Drug.csv")

drugsubset=["naproxen","clarithromycin","chloroquine","kaletra","Favipiravir","Avigan",'hydroxychloroquine','baricitinib']
Make2DPlot(dat_joined_vd[dat_joined_vd.word_drug.isin(drugsubset)],"virus","drug")
Make2DPlot(dat_joined_vd[dat_joined_vd.word_drug.isin(drugsubset)],"virus","drug",single_sentence_plots=True)
# Prune and join, and extract overlap counts
dat_joined_dt=dat_drugs.join(dat_therapies, rsuffix='_therapy',lsuffix="_drug")
dat_joined_dt=dat_joined_dt[dat_joined_dt.notna().word_drug & dat_joined_dt.notna().word_therapy]

dat_joined_dt=dat_joined_dt.drop(["sha_drug","blockid_drug","sec_drug"],axis=1).reset_index().rename(columns={"sha_therapy":"sha","blockid_therapy":"blockid","sec_therapy":"sec"})
SingleSentence=[]
for i in dat_joined_dt.index:
    SingleSentence.append(SameSentenceCheck(dat_joined_dt.block[i],dat_joined_dt.pos_drug[i],dat_joined_dt.pos_therapy[i]))
dat_joined_dt.insert(len(dat_joined_dt.columns),'same_sentence',SingleSentence)
dat_joined_dt.to_csv("Overlaps_Drug_Therapy.csv")
Make2DPlot(dat_joined_dt[dat_joined_dt.word_drug.isin(drugsubset)],"drug","therapy")
Make2DPlot(dat_joined_dt[dat_joined_dt.word_drug.isin(drugsubset)],"drug","therapy",single_sentence_plots=True)
# Prune and join, and extract overlap counts
dat_joined_de=dat_drugs.join(dat_exps, rsuffix='_exp',lsuffix="_drug")
dat_joined_de=dat_joined_de[dat_joined_de.notna().word_drug & dat_joined_de.notna().word_exp]

dat_joined_de=dat_joined_de.drop(["sha_drug","blockid_drug","sec_drug"],axis=1).reset_index().rename(columns={"sha_exp":"sha","blockid_exp":"blockid","sec_exp":"sec"})
SingleSentence=[]
for i in dat_joined_de.index:
    SingleSentence.append(SameSentenceCheck(dat_joined_de.block[i],dat_joined_de.pos_drug[i],dat_joined_de.pos_exp[i]))
dat_joined_de.insert(len(dat_joined_de.columns),'same_sentence',SingleSentence)
dat_joined_de.to_csv("Overlaps_Drug_Experiment.csv")
Make2DPlot(dat_joined_de[dat_joined_de.word_drug.isin(drugsubset)],"drug","exp")
Make2DPlot(dat_joined_de[dat_joined_de.word_drug.isin(drugsubset)],"drug","exp",single_sentence_plots=True)
dat_joined_vtd=dat_therapies.join(dat_viruses, rsuffix='_virus',lsuffix="_therapy").join(dat_drugs)
dat_joined_vtd=dat_joined_vtd[dat_joined_vtd.notna().word_therapy & dat_joined_vtd.notna().word_virus & dat_joined_vtd.notna().word]
grouped_vtd=dat_joined_vtd.groupby(['word_therapy','word_virus','word'])
grouped_vtd.count().sha_therapy
dat_joined_vtd=dat_joined_vtd.reset_index().drop(['sha_therapy','blockid_therapy','sec_therapy','sha_virus','blockid_virus','sec_virus'],axis=1).rename(columns={'word':'word_drug','pos':'pos_drug'}).set_index('sha')
dat_joined_vtd=dat_joined_vtd[["block","sec","blockid","word_therapy","pos_therapy","word_virus", "pos_virus","word_drug","pos_drug"]]
dat_joined_vtd.to_csv("Overlaps_Drug_Therapy_Virus.csv")

OverlapsVirus=pd.read_csv("./Overlaps_Virus_Drug.csv")
OverlapsTherapy=pd.read_csv("./Overlaps_Drug_Therapy.csv")

PapersWithVirusDrugOverlap=OverlapsVirus.sha.unique()
PapersWithVirusMention=dat_viruses.sha.unique()
OverlapsTherapy=OverlapsTherapy[OverlapsTherapy.same_sentence==1]

# This two helper function does its best to extract the year from the 
#  inconsistently formatted metadata

def ConvertDateToYear(datestring):
    import dateutil.parser as parser

    if(pd.notna(datestring)):
        try:
            date=parser.parse(str(datestring),fuzzy=True)
            return date.year
        except ValueError:
            return 0
    else:
        return 0
    


# Take the elements we need out of the paper metadata
meta=pd.read_csv("/kaggle/input/CORD-19-research-challenge/metadata.csv")
years=meta.publish_time.apply(ConvertDateToYear)
meta.insert(len(meta.columns),'year',years)
meta_to_use=meta.set_index('sha')[['doi','title','year','abstract']]

# And mix it in
OverlapsTherapy=OverlapsTherapy.set_index('sha').join(meta_to_use).reset_index()
# Extract the relevant sentences where matches were found
SentenceEnders="\. |; \! \? "
ExtractedSentences=[]
for i in OverlapsTherapy.index:
    sentences = re.split(SentenceEnders,OverlapsTherapy.block.loc[i])
    RunningCount=0
    ExtractedSentences.append(np.NaN)

    for s in range(0,len(sentences)):
        RunningCount=RunningCount+len(sentences[s])
        if(OverlapsTherapy.pos_drug.loc[i]<RunningCount):
            ExtractedSentences[-1]=sentences[s]
            break
            
OverlapsTherapy.insert(len(OverlapsTherapy.columns),'sentence',ExtractedSentences)
# Check for coincidences in block and paper
CoronaInPaper=OverlapsTherapy.sha.isin(PapersWithVirusMention)
CoronaInBlock=OverlapsTherapy.sha.isin(PapersWithVirusDrugOverlap)
OverlapsTherapy.insert(len(OverlapsTherapy.columns),'corona_paper',CoronaInPaper)
OverlapsTherapy.insert(len(OverlapsTherapy.columns),'corona_block',CoronaInBlock)
#Tidy and trim
OverlapsTherapy=OverlapsTherapy.rename(columns={'word_drug':'drug','sec':'section','block':'paragraph'}).drop(['blockid','pos_drug','pos_therapy','Unnamed: 0','same_sentence','word_therapy'],axis=1)
# Remove a few obvious fakes
Fakes=['injection','glucose','perform','ethanol','methanol','paraffin','soybean','horseradish','ginger','mouthwash','oregano','formaldehyde','alcohol']
OverlapsTherapy=OverlapsTherapy[ OverlapsTherapy.drug.isin(Fakes)==False]

# This is the final file that is used as input to the visualization stage
OverlapsTherapy.to_csv("DrugVisData.csv")
data = OverlapsTherapy
output_data = data
out_dir = './'


# negated term list (use the human annotated version)
neg_list = pd.read_csv('/kaggle/input/neg-list-complete/neg_list_complete.txt', sep='\t', header=0)
neg = neg_list['ITEM'].values
neg_term = [' ' + item + ' ' for item in neg]
neg_term.extend(item + ' ' for item in neg)


for i in range(0,len(data)):
    if pd.isnull(data.loc[i,'sentence']):
        output_data.loc[i,'Is_Negated'] = 0
    else:
        # tag negated or affirmed based on string matching --- negation term list
        # add one space to prevent loss of 'no ', 'not ', ... etc.
        if any(substring in ' ' + data.loc[i,'sentence'].lower() for substring in neg_term):
            output_data.loc[i,'Is_Negated'] = 1
        else:
            output_data.loc[i,'Is_Negated'] = 0

# save results in a output file
output_data.to_csv('DrugVisData_Negated_Output.csv',index=False)
negated_drug_mentions = output_data.loc[output_data.Is_Negated==1,'drug']\
                                    .groupby(output_data['drug'])\
                                    .value_counts()\
                                    .droplevel(level=0)
print('Top 20 most negated drugs:\n')
print(negated_drug_mentions.nlargest(20))
asserted_drug_mentions = output_data.loc[output_data.Is_Negated==0,'drug']\
                                    .groupby(output_data['drug'])\
                                    .value_counts()\
                                    .droplevel(level=0)
print('Top 20 most asserted drugs:\n')
print(asserted_drug_mentions.nlargest(20))
drug_mentions = output_data.groupby([output_data['drug'],output_data.Is_Negated])\
                            .size().to_frame(name = 'size').reset_index()\
                            .pivot(index='drug',columns='Is_Negated',values='size').fillna(0).reset_index()

drug_mentions['Percentage Negations'] = (drug_mentions[1]*100)/(drug_mentions[0]+drug_mentions[1])

drug_mentions.hist(column='Percentage Negations')
# Drugs with 100% negation
drug_mentions.nlargest(n=1,columns='Percentage Negations',keep='all').plot.bar('drug',[1,0],figsize=(15,6))
# Drugs with 50% negation
drug_mentions.loc[drug_mentions['Percentage Negations']==50,:].plot.bar('drug',[1,0],figsize=(15,6))
