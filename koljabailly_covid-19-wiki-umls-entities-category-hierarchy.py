!pip install scispacy
!pip install spacy
import csv
import logging
import json
import sys
import requests
import spacy
import scispacy
from oauthlib.common import urlencode

# Author: Kolja Bailly

# This script use the article metadata of the COVID-19 open reasearch dataset and extend it with additional information
# - general topics of article abstracts
# - key statements for every article abstract in the form subject-predicate-object
# - Named Entities from article abstracts in a controlled vocabular (the unified medical language system UMLS)
# - a hierarchical category tree for every abstract and statement from the UMLS Ontology
#
# The results are written into a extended metadata csv file and can optionally exported into a semantic mediawiki pageFile format
# The goal is to use the extended information to provide a ordered category tree within a semantic wiki frontend with all articles and statements sorted into categories in a controlled vocabular
# A wiki containing this data can be accessed via: http://www.hesinde.org

# The extended metadata result csv file is included in the data section of this notebook. Feel free to use it for your own purposes.
# The code doesnt run within kaggle currently due to a failure in installing scispacy library.
# Run the code locally instead, don't forget to insert your IBM Watson API Key in class 'hesinde' below before running it 

#the following methods are defined below:
# 1. readMetadata():
# - read articles metadata row by row from the file metadata.csv until given limit is reached 
# - create a unique article nr and id, matching the requirements of mediawiki pages
# - a blacklist of articles may given as parameter to avoid processing of already annotated articles
#
# 2. writeExtendedMetadata()
# - write annotated article metadada to csv
# - option to rewrite whole file or to append to existing file 
#
# 3. readExtendedMetadata()
# - read in extended metadata from file (for example to add articles to blacklist for a new processing run)
#
# 4. initWatsonData
# - connect to IBM Cloud Natural Language Processing API 
# - retrieve general topics for each article abstract 
# - retrieve key statements from each article abtstract in the form subject-predicate-object
#
# 5. InitNLPData()
# - Use SciSpacy Library for Named Entity Recognition (NER) of article abstracts
# - Use SciSpacy to translate recognized enities into the Unified Medical Language System (UMLS) vocabular
# - Save semantic parent trees for each article abstract and each watson statement using method 'getSemanticCategories'
#
# 6. getSemanticCategories()
# - Use SciSpacy to get a subTree of the UMLS Semantic type tree (from root to entity node)
#  
# 7. sanitizeMediaWikiPageName()
# - change given string to mach requirements of mediawiki pagenames
#
# 8. createWikiPageFiles()
# - use extended metadata to create a mediawiki conform page file (to import extended metadata and categories as pages into a mediawiki instance)
#
# ----Class Hesinde----
# - wrapper class for running the above methods
# - create extended metadata
# - create mediawiki pageFiles
#

# known bugs:
# in wikiPageFiles, statements may contain invalid characters in the article link field relatedToArticle (doesnt meet the requirements of a mediawiki pageName).
# these names have to be sanitized as well as the article page names.

##################################
# Read in article metadata File
##################################
def readMetadata(articles, skipArticleIds, limit):
    with open('metadata.csv') as csv_file:
    #with open('testRoutine.csv') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        line_count = 0
        articles_added = 0
        firstContentLine = 1
        maxArticles = limit
        colNames = []
        for row in csv_reader:
            if (articles_added < maxArticles or maxArticles == -1):
                if line_count < firstContentLine:
                    print(f'header: = {row} ')
                    colNames = row
                else:

                    #read article data
                    actArticle = {}
                    for i in range(len(row)):
                        actArticle[colNames[i]] = row[i]
                    # create page name for wiki
                    if ( actArticle['sha']+"-"+actArticle['title'] in skipArticleIds):
                        print(f"skip rowNr {line_count} article title: {actArticle['title']}")
                        line_count += 1
                        continue
                    actArticle['nr'] = len(skipArticleIds) + articles_added + 1  # begin at 1
                    actArticle['id'] = actArticle['nr'].__str__() + "-" + actArticle['title'].replace("[","(").replace("]",")").replace("#","").replace("<","").replace(">","").replace("|","").replace("{","").replace("}","").replace("_","")
                    # check for skip Articles

                    actArticle['semanticCategories'] = []
                    if( actArticle['abstract']  ):
                        articles.append( actArticle )
                        articles_added += 1
            line_count += 1
        print(f'Processed {line_count} lines.')
        print(f'Articles added {articles_added} .')

def writeExtendedMetadata(articles, semanticTreeNodes, append):
    openType = 'w'
    if( append ):
        openType ='a'
    with open('metadata_ext.csv', openType, newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        if( not append ):
            writer.writerow(articles[0])
        for article in articles:
            actRow = [
                article["sha"],
                article["source_x"],
                article["title"],
                article["doi"],
                article["pmcid"],
                article["pubmed_id"],
                article["license"],
                article["abstract"],
                article["publish_time"],
                article["authors"],
                article["journal"],
                article["Microsoft Academic Paper ID"],
                article["WHO #Covidence"],
                article["has_full_text"],
                article["full_text_file"],
                article["nr"],
                article["id"],
                json.dumps(article["semanticCategories"]),
                json.dumps(article["watsonResult"])
            ]
            writer.writerow( actRow )

def readExtendedMetadata(articles, semanticTreeNodes):

    try:
        with open('metadata_ext.csv') as csv_file:
            csv.field_size_limit(sys.maxsize)
            csv_reader = csv.reader(csv_file, delimiter=',')

            line_count = 0
            articles_added = 0
            firstContentLine = 1
            maxArticles = -1
            colNames = []
            for row in csv_reader:
                if articles_added < maxArticles or maxArticles == -1:
                    if line_count < firstContentLine:
                        print(f'header: = {row} ')
                        colNames = row
                    else:
                        actArticle = {}
                        for i in range(len(row)):
                            if( colNames[i] == "semanticCategories" ):
                                arrayStr = row[i]
                                arrayStr = arrayStr.replace("[","").replace("]","").replace("'","").replace('"','')
                                actArticle[colNames[i]] = arrayStr.split(", ")
                            elif( colNames[i] == "watsonResult"):
                                #parse dictionary with json
                                actArticle[colNames[i]] = json.loads( row[i] )
                            else:
                                actArticle[colNames[i]] = row[i]
                        #print(actArticle['id'])
                        # create page name for wiki
                        articles.append(actArticle)
                        articles_added += 1
                line_count += 1
            print(f'Processed {line_count} lines.')
            print(f'Articles added {articles_added} .')
    except IOError:
        print("File not accessible")


##########################################
# connect watson service
##########################################
def initWatsonData( articles, apiKey):

    url = 'https://api.eu-de.natural-language-understanding.watson.cloud.ibm.com/instances/b00318ad-c96a-4fea-96e5-94f27297002f/v1/analyze'
    headers = { 'Content-Type': 'application/json' }
    params = {'version': '2018-11-16'}
    auth = ('apikey', apiKey)
    articlesToDelete = []
    i = 0
    for article in articles:
        # get watson data (/concepts)
        data = { "features": {
                "semantic_roles": {},
                "concepts":{}
                },
                "text": article['abstract']
            }

        response = requests.post(url, headers=headers, auth=auth, json=data, params=params)
        #print(response.request.body)
        #print(response.request.headers)
        if response.status_code == 200:
            print(f'watson request {i+1}/{len(articles)} received')
            #print(response.content)
        else:
            print("error in watson request")
            print(response.content)
        #convert response to json
        jsonResult = json.loads(response.content)

        #append watson response data to article object
        article['watsonResult'] = {}
        if( 'concepts' in jsonResult and 'semantic_roles' in jsonResult ):
            article['watsonResult']['concepts'] = jsonResult['concepts']
            article['watsonResult'][''] = jsonResult['semantic_roles']
        else:
            articlesToDelete.append(article)

        i += 1
    # remove rticles with missing data
    for toDelete in articlesToDelete:
        articles.remove(toDelete)

    # init a list of categroytreeNodes to save all the semantic categorys and their tree structure into
    # key=categoryName values = List of parent Category Names
    semanticCategoryTreeNodes = {}


####################################
#use sciSpacy to detect entities
####################################
def getSemanticCategories( doc, linker, listToAddCategories, semanticCategoryTreeNodes):
    for entity in doc.ents:  # go through all identified medical entities
        for umls_ent in entity._.umls_ents:  # get all matched concepts in umls ontology (normally there is only one match)
            umlsEntity = linker.umls.cui_to_entity[umls_ent[0]]
            listToAddCategories.append(
                umlsEntity.canonical_name)  # add umls entity name as category to article

            # now add all semantic type categories of this entity as parent categories
            entityCategory = linker.umls.semantic_type_tree.get_node_from_id(umlsEntity.types[0])
            semanticCategoryTreeNodes[umlsEntity.canonical_name] = entityCategory.full_name

            # now go up the semantic type tree and add all nodes to the category tree
            semanticTreeNode = entityCategory
            for i in range(semanticTreeNode.level):
                semanticTreeNodeParent = linker.umls.semantic_type_tree.get_parent(semanticTreeNode)
                # save node in category tree if not exist, set his parent as value
                if (semanticTreeNode.full_name not in semanticCategoryTreeNodes):
                    parentCatName = semanticTreeNodeParent.full_name
                    if (parentCatName == "UnknownType"):
                        parentCatName = "UMLS"
                    semanticCategoryTreeNodes[semanticTreeNode.full_name] = parentCatName
                # set act parent as base node and go up the tree
                semanticTreeNode = semanticTreeNodeParent

def initNLPdata( articles, semanticCategoryTreeNodes):
    print("load sciSpacy NLP Libraries...")

    from scispacy.umls_linking import UmlsEntityLinker

    # load NLP model
    nlp = spacy.load("en_core_sci_lg")
    # load linker for entity recognition in unified medical language system (UMLS)
    linker = UmlsEntityLinker(resolve_abbreviations=True)
    nlp.add_pipe(linker)

    print("start NLP process...")
    # init category tree
    for article in articles:
        # do NLP on abstract
        doc = nlp( article['abstract'] )
        # add categories to artcile
        getSemanticCategories(doc, linker, article['semanticCategories'], semanticCategoryTreeNodes)

        # add categories to  (predicates obtained by watson) of article
        if( 'watsonResult' in article ):
            if( '' in article['watsonResult'] ):
                for relation in article['watsonResult']['']:
                    if ('sentence' in relation):
                        doc2 = nlp( relation['sentence'] )
                        relation['semanticCategories'] = []
                        getSemanticCategories(doc2, linker, relation['semanticCategories'],semanticCategoryTreeNodes)

    print( list(semanticCategoryTreeNodes) )
    

def sanitizeMediaWikiPageName(pageName):
    return pageName.replace("[", "(").replace("]", ")").replace("#", "").replace("<","").replace(">", "").replace("|", "").replace("{", "").replace("}", "").replace("_", "")

def createWikiPageFiles(path, articles, semanticCategoryTreeNodes):
    categoryPath="Category/"
    statementPath="Statement/"
    # create Category Pages
    baseCatName = "Generated Categories"
    watsonCategoryName = "Topics"
    umlsCategoryName = "UMLS"

    F = open(path + categoryPath + baseCatName, "w")
    F.close()

    F = open(path + categoryPath + watsonCategoryName, "w")
    F.write("[[Category:"+baseCatName + "]]")
    F.close()

    F = open(path + categoryPath + umlsCategoryName, "w")
    F.write("[[Category:" + baseCatName + "]]")
    F.close()

    articleNr = 0
    for article in articles:
        articleNr += 1
        if( articleNr % 100 == 0 ):
            print( f"{articleNr}/{len(articles)}({round((articleNr/len(articles)),1)}%)" )
        fileName = article['id']
        for concept in article['watsonResult']['concepts']:
            categoryName = "Topic_" + concept['text'].replace("/","|") # slash is forbidden in filenames
            actContent = "[[Category:" + watsonCategoryName + "]]"  # parent category
            F = open(path + categoryPath + categoryName, "w")
            F.write(actContent)
            F.close()

        # add UMLS concepts as categories
        for categoryName in list(semanticCategoryTreeNodes):
            categoryName_sanitized = sanitizeMediaWikiPageName(categoryName)
            actContent = "[[Category:{}]]".format(sanitizeMediaWikiPageName(semanticCategoryTreeNodes[categoryName]).replace('"',''))  # add parent as category
            F = open(path + categoryPath + categoryName_sanitized, "w")
            F.write(actContent)
            F.close()

        articleConceptTemplate = """{{{{Article
                |unique_nr={}
                |sha={}
                |source={}
                |title={}
                |doi={}
                |pmcid={}
                |pubmed_id={}
                |license={}
                |abstract={}
                |publish_time={}
                |authors={}
                |journal={}
                |microsoftAcademicPaperId={}
                |WHO_CovidenceNr={}
                |hasFullText={}
                |full_text_file={}
                }}}}

                {}"""
        # add act article to categories
        categories = ""
        for categoryName in article['semanticCategories']:
            categoryName = sanitizeMediaWikiPageName(categoryName).replace('"', "")
            #print("assign umls category:" + categoryName + " to article:" + article['id'])
            categories = categories + "[[Category:" + categoryName + "]]"
        for concept in article['watsonResult']['concepts']:
            categoryName = "Topic_" + concept['text']
            #print("assign watson category:" + categoryName + " to article:" + article['id'])
            categories = categories + "[[Category:" + categoryName + "]]"

        articleName = article['id'].replace("/","|") # slash is forbidden in filenames
        if (len(articleName) > 255):
            articleName = articleName[0:255]

        articleContent = articleConceptTemplate.format(
            article["nr"],
            article["sha"],
            article["source_x"],
            article["title"],
            article["doi"],
            article["pmcid"],
            article["pubmed_id"],
            article["license"],
            article["abstract"],
            article["publish_time"],
            article["authors"],
            article["journal"],
            article["Microsoft Academic Paper ID"],
            article["WHO #Covidence"],
            article["has_full_text"],
            article["full_text_file"],
            categories)

        F = open(path + articleName, "w")
        F.write(articleContent)
        F.close()

        # insert article result statements (Natural language understanding)
        articleResultTemplate = """{{{{Statement
                    |StatementSubject={}
                    |StatementPredicate={}
                    |StatementObject={}
                    |StatementType={}
                    |StatementSourceText={}
                    |StatementRelatedTo={}
                    |publish_time={}
                    }}}}

                    {}"""

        statementNr = 1
        for relation in article['watsonResult']['']:
            categories = ""
            if ('subject' in relation and 'action' in relation and 'object' in relation):
                if ('text' in relation['subject'] and 'normalized' in relation['action'] and 'text' in relation['object']):
                    if ('semanticCategories' in relation):
                        for categoryName in relation['semanticCategories']:
                            categoryName = sanitizeMediaWikiPageName(categoryName)
                            categories = categories + "[[Category:" + categoryName + "]]"
                    # insert Natural Language Understanding (relation)triples as results of articles
                    # articleResultStatementPageName = f"Statement:Result{statementNr}ofArticleNr{article['nr']}"
                    articleResultStatementPageName = f"{relation['subject']['text']}--{relation['action']['normalized']}--{relation['object']['text']}"
                    if (len(articleResultStatementPageName) > 235):
                        articleResultStatementPageName = articleResultStatementPageName[0:235]
                    articleResultStatementPageName = sanitizeMediaWikiPageName(
                        articleResultStatementPageName + f"...ArticleNr{article['nr']}")
                    articleResultStatementPageName = articleResultStatementPageName.replace("/","|") # slash is forbidden in filenames
                    actContent = articleResultTemplate.format(
                        relation['subject']['text'],
                        relation['action']['normalized'],
                        relation['object']['text'],
                        "NaturalLanguageUnderstandingResult",
                        relation['sentence'],
                        article['id'],
                        article['publish_time'],
                        categories)
                    F = open(path + statementPath + articleResultStatementPageName, "w")
                    F.write(actContent)
                    F.close()
                    statementNr += 1
                    
class Hesinde:
    logging.basicConfig(level=logging.WARNING)
    articles = []
    semanticCategoryTreeNodes = {}

    #parameter
    rewriteAllCategories = False
    path="wikiPages/"
    watsonApiKey="<YourWatsonAPIKey>" # see comment below how to get an api key...
    # 1) register at ibm cloud for free account: 
    # https://cloud.ibm.com/registration?target=%2Fcatalog%2Fservices%2Fnatural-language-understanding%3FhideTours%3Dtrue%26&cm_sp=WatsonPlatform-WatsonPlatform-_-OnPageNavCTA-IBMWatson_NaturalLanguageUnderstanding-_-Watson_Developer_Website
    # 2) create lite(free) plan for natural language understanding
    # 3) get your api key, see: https://cloud.ibm.com/docs/iam?topic=iam-userapikey
    
    def __init__(self):
       # while(True):
        self.createExtendedMetadata() # create first 2000 extended metadata
        #self.articles=[]
        #self.semanticCategoryTreeNodes = {}
        #self.readExtendedMetadataAndinsertIntoMediaWiki()

    def createExtendedMetadata(self):
        skipExistingExtendedMetadataEntries = True #when reading data for metadata extension, shall existing articles skipped instead of rewritten?
        appendOnExistingMetadata = True # when saving extended metadata, shall new lines appended on exiting file instead of replacing them?

        skipArticleIds = []
        skipArticles = []
        if(skipExistingExtendedMetadataEntries):
            readExtendedMetadata( skipArticles, {} )
            for article in skipArticles:
                skipArticleIds.append( article['sha']+"-"+article['title'] )
        readMetadata(self.articles, skipArticleIds, 20)
        initWatsonData(self.articles)
        initNLPdata(self.articles, self.semanticCategoryTreeNodes)
        writeExtendedMetadata(self.articles, self.semanticCategoryTreeNodes, appendOnExistingMetadata)

    def readExtendedMetadataAndinsertIntoMediaWiki(self):
        # fill object attributes with data from file
        readExtendedMetadata(self.articles, self.semanticCategoryTreeNodes)
        createWikiPageFiles(self.path, self.articles, self.semanticCategoryTreeNodes)
        #insertDataIntoMediaWiki(self.articles, self.semanticCategoryTreeNodes, self.rewriteAllCategories)



####### start script
c = Hesinde()     