from pymed import PubMed



import pandas as pd

# Create a PubMed object that GraphQL can use to query

# Note that the parameters are not required but kindly requested by PubMed Central

# https://www.ncbi.nlm.nih.gov/pmc/tools/developers/

pubmed = PubMed(tool="MyTool", email="my@email.address")



# Create a GraphQL query in plain text

query = '(amyotrophic lateral sclerosis) AND ("1976"[Date - Publication] : "3000"[Date - Publication])'



results = list(pubmed.query(query, max_results=25000))



article_dict = {}

article_dict_list = []

als_articles = pd.DataFrame()



# Loop over the retrieved articles

for i, article in enumerate(results):

    article_dict = {}

    

    if not article.abstract:

        pass 

    

    else:

        article_dict['abstract'] = article.abstract

        

        try:

            article_dict["conclusions"]= article.conclusions

        except:

            article_dict["conclusions"]= 'none'

            

        article_dict["copyrights"] = article.copyrights

        article_dict["doi"] = article.doi

        

        try:

            article_dict["journal"] = article.journal

        except:

            article_dict["journal"] ='none'

            

        try:

            article_dict["methods"] = article.methods

        except:

            article_dict["methods"]='none'

        article_dict["publication_date"] = article.publication_date

        article_dict["pubmed_id"]= article.pubmed_id

        

        try:

            article_dict["results"]=article.results

        except:

            article_dict["results"]='none'

        article_dict["title"] = article.title



        try:

            article_dict["keywords"] = '; '.join(article.keywords) 

        except:

            article_dict["keywords"] = "none"



        authors = []

        affils = []

        for a,author in enumerate(article.authors):

            auth = []

            auth.append(author['lastname'])

            auth.append(author['firstname'])

            auth.append(author['initials'])

            author_name = ', '.join(str(x) for x in auth)

            authors.append(author_name)



            try:

                if author['affiliation']!= None:

                    affils.append(author['affiliation'])

                else:

                    affils.append('none listed')

            except:

                affils.append('none listed')





        article_dict["authors"] = '; '.join(authors) 

        article_dict["affiliations"] = '; '.join(affils) 





        article_dict_list.append(article_dict)

        #print(article.toJSON())

    

als_articles = pd.concat([pd.DataFrame(article_dict_list)])



als_articles.head()

len(als_articles)



als_articles.to_csv('./als_articles.csv', encoding='utf8', index=False)