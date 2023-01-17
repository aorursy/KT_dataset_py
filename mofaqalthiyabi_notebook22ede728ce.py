import json
import os, glob

#directory for atricle json files:
articles_dir = '/Users/Mofaq/Desktop/comm_use_subset'
#output directory for processed files:
output_dir = '/Users/Mofaq/Desktop/s'

os.chdir(articles_dir)
for file in glob.glob('*.json'):
    print('Processing file: ',file)
    with open(file, 'r', encoding = 'utf8') as article_file:
        article = json.load(article_file)
        title = article['metadata']['title']
        abstract_sections = []
        abstract_texts = dict()
        
        body_sections = []
        body_texts = dict()

        #reading abstracts
        for abst in article['abstract']:
            abst_section = abst['section']
            abst_text = abst['text']
            if abst_section not in abstract_sections:
                abstract_sections.append(abst_section)
                abstract_texts[abst_section] = abst_text
            else:
                abstract_texts[abst_section] = abstract_texts[abst_section] + '\n' + abst_text
        
        #reading body
        for body in article['body_text']:
            body_section = body['section']
            body_text = body['text']
            if body_section not in body_sections:
                body_sections.append(body_section)
                body_texts[body_section] = body_text
            else:
                body_texts[body_section] = body_texts[body_section] + '\n' + body_text
        
        with open(output_dir+'/clean.'+file.replace('.json','.txt') , 'w', encoding = 'utf8') as out_file:
            out_file.writelines(title)
            out_file.writelines('\n\n')
            #print abstracts
            for a_section in abstract_sections:
                out_file.writelines('\n\n')
                out_file.writelines(a_section)
                out_file.writelines('\n')
                out_file.writelines(abstract_texts[a_section])
            #print body
            for b_section in body_sections:
                out_file.writelines('\n\n')
                out_file.writelines(b_section)
                out_file.writelines('\n')
                out_file.writelines(body_texts[b_section])
            out_file.writelines('\n')