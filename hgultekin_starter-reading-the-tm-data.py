import pandas as pd
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
CD = '../input/paralel-translation-corpus-in-22-languages/' #initially there were 22 languages, two more added afterwards :)
SL = 'EN' #this is a constant and should not be changed, i.e. Source Language is always English
TL = 'DE' #depending on the desired Target Language, this could be set, available abbr. choices are written in the introduction paragraph
pd.read_csv(CD+SL+'-'+TL+'/'+SL+'-'+TL+'.txt', sep='\t', header = None)[[0,1]].rename(columns = {0:SL, 1:TL})
import pandas as pd
from xml.dom import minidom
def process_tuv(tuv):
    lang = tuv.getAttribute("lang")
    if lang == '':
        lang = tuv.getAttribute("xml:lang")
    seg = tuv.getElementsByTagName('seg')[0]
    txt = seg.childNodes[0].data
    return lang, txt

def read_tmx_files(path):

    """Read function takes in a path to TMX translation file and outputs the metadata and a pandas dataframe.

    Args:
        param1 (str): The path to the TMX translation file

    Returns:
        dict: The header of the TMX file, which contains metadata
        DataFrame: A Pandas Dataframe. Each line item consists of source_language, source_sentence, target_language, target_sentence

    """
    # parse an xml file by name
    tmx = minidom.parse(path)

    # Get metadata
    metadata = {}
    header = tmx.getElementsByTagName('header')[0]
    for key in header.attributes.keys():
        metadata[key] = header.attributes[key].value
        
    srclang = metadata['srclang']

    # Get translation sentences
    body = tmx.getElementsByTagName('body')[0]
    translation_units = body.getElementsByTagName('tu')
    items = []
    count_unpaired = 0
    for tu in translation_units:
        if len(tu.getElementsByTagName('tuv')) < 2:
            print("Unpaired translation. Ignoring...")
            count_unpaired = count_unpaired + 1
        else:
            srclang, srcsentence = process_tuv(tu.getElementsByTagName('tuv')[0])
            targetlang, targetsentence = process_tuv(tu.getElementsByTagName('tuv')[1])
            item = {
                'source_language': srclang,
                'source_sentence': srcsentence,
                'target_language': targetlang,
                'target_sentence': targetsentence
            }
            items.append(item)

    df = pd.DataFrame(items)
    if count_unpaired > 0:
       print("The data contained %d unpaired translations which were ignored" % (count_unpaired))
    return metadata, df
metadata, df = read_tmx_files('../input/paralel-translation-corpus-in-22-languages/Sample TMX file (EN-GA)/EN-GA.tmx')
df.head()