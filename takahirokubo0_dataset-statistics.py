from pathlib import Path


data_folder = Path.cwd().parent.joinpath("input/chabsa-dataset/chABSA-dataset/")

def check_data_existence(folder):
    file_count = len(list(folder.glob("e*_ann.json")))
    if  file_count == 0:
        raise Exception("Processed Data does not exist.")
    else:
        print("{} files exist.".format(file_count))


check_data_existence(data_folder)
import json
import pandas as pd


companies = []
sentences = []
entities = []


for f in data_folder.glob("e*_ann.json"):
    with f.open(encoding="utf-8") as j:
        d = json.load(j)
        
        # company infos
        company_info = d["header"]
        companies.append(company_info)
        
        # sentences
        company_code = company_info["document_id"]
        for s in d["sentences"]:
            line = {
                "company": company_code,
                "sentence": s["sentence"],
                "entities": len(s["opinions"])
            }
            sentences.append(line)

            # entities
            for o in s["opinions"]:
                entities.append(o)


companies = pd.DataFrame(companies)
sentences = pd.DataFrame(sentences)
entities = pd.DataFrame(entities)
companies.head(5)
sentences.head(5)
entities.head(5)
%matplotlib inline
translation = """
水産・農林業	Fishery, Agriculture & Forestry
鉱業	Mining
建設業	Construction
食料品	Foods
繊維製品	Textiles and Apparels
パルプ・紙	Pulp and Paper
化学	Chemicals
医薬品	Pharmaceutical
石油・石炭製品	Oil and Coal Products
ゴム製品	Rubber Products
ガラス・土石製品	Glass and Ceramics Products
鉄鋼	Iron and Steel
非鉄金属	Nonferrous Metals
金属製品	Metal Products
機械	Machinery
電気機器	Electric Appliances
輸送用機器	Transportation Equipment
精密機器	Precision Instruments
その他製品	Other Products
電気・ガス業	Electric Power and Gas
陸運業	Land Transportation
海運業	Marine Transportation
空運業	Air Transportation
倉庫・運輸関連業	Warehousing and Harbor Transportation
情報・通信業	Information & Communication
卸売業	Wholesale Trade
小売業	Retail Trade
銀行業	Banks
証券、商品先物取引業	Securities and Commodities Futures
保険業	Insurance
その他金融業	Other Financing Business
不動産業	Real Estate
サービス業	Services
"""

translation_list = [t.split("\t") for t in translation.split("\n") if t]
translation_list = dict(translation_list )
companies["category33_en"]  = companies["category33"].apply(lambda c: translation_list[c])
companies.groupby(["category33_en"]).count()["edi_id"].sort_values(ascending=False).plot(kind="bar", figsize=(15,5))
print("{} entities are annotated.".format(len(entities)))
entities.groupby(["category"]).count()["target"].sort_values(ascending=False).plot(kind="bar")
(entities.groupby(["category"]).count()["target"].sort_values(ascending=False).cumsum() * 100 / len(entities)).plot.line(secondary_y=True, style="g", rot=90)
entities.groupby(["polarity"]).count()["target"].plot.bar()
entities.groupby(["polarity", "category"]).count()["target"].divide(entities.groupby(["category"]).count()["target"]).unstack("polarity").plot.bar(stacked=True)
print("The sentences that have entities are {}.".format(len(sentences[sentences["entities"] > 0])))
print("The number of sentences are {}.".format(len(sentences)))
sentences[sentences["entities"] > 0].groupby(["entities"]).count()["company"].plot.bar()