from zipfile import ZipFile



with ZipFile('submission.zip','w') as zip:           

  zip.write('../input/landmark-retrieval-model/model/saved_model.pb', arcname='saved_model.pb') 

  zip.write('../input/landmark-retrieval-model/model/variables/variables.data-00000-of-00001', arcname='variables/variables.data-00000-of-00001') 

  zip.write('../input/landmark-retrieval-model/model/variables/variables.index', arcname='variables/variables.index') 