import time
import lxml.etree as ET
import pandas as pd


def xml_parsing_corpus_reddit(data):
    return pd.DataFrame(
        data=[node.text.strip() for node in data.xpath("//dialog/s/utt")],
        columns=["text"]
    )


def parsing_xml_file(file_path):
    #Initializes the parser
    parser = ET.XMLParser(recover=True)
    #Parses the file
    tree = ET.parse(file_path, parser=parser)
    xroot = tree.getroot()
    return xroot


# path to file
file_path = "../input/french-reddit-discussion/final_SPF_2.xml"
output_path = "reddit_text_corpus.csv"

start = time.time()

print("Parsing xml ...")
xroot = parsing_xml_file(file_path)
print("Done")

df = xml_parsing_corpus_reddit(xroot)

end = time.time()
print(f"Processing time: {end - start}")
print(df)

print(f"Saving to csv at: {output_path}")
df.to_csv(output_path, sep=";", index=False)
