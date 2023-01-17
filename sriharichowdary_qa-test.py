!pip install allennlp
from allennlp.predictors.predictor import Predictor as AllenNLPPredictor
pred=AllenNLPPredictor.from_path(
            "https://storage.googleapis.com/allennlp-public-models/bidaf-elmo-model-2018.11.30-charpad.tar.gz"
        )
passage = "Nellore is a city on the banks of Penna River,located in Nellore district of the Indian state of Andhra Pradesh.It serves as the headquarters of the district, as well as Nellore mandal and Nellore revenue division. It is the fourth most populous city in the state. It is at a distance of 275 km from the capital city of Andhra Pradesh. Nellore City accounts for 65% of the urban population of Nellore District.It has an average elevation of 18 metres (59 ft). Nellore had been under the rule of Cholas, Pallavas, pandyas, Maurya Dynasty, Kharavela of Chedi dynasty, Satavahanas, Kakatiyas, Eastern Gangas of Kalinga Empire, Vijayanagara Empire, Arcot Nawabs and other dynasties.\
Nellore existed from the times of the Cholas ruled by Rajendra Chola I under Tanjavur Mauryan empire and was ruled by Ashoka in the 3rd century B.C. Nellore was conquered by the Rulers of the Pallava Dynasty and it was under their till the 6th century AD, subsequently the Chola rulers ruled Nellore for a long period of time. The Telugu Cholas met their decline in the 13th Century.Tamil inscriptions indicate that it formed part of Chola kingdom till their decline in the thirteenth century AD It later became a part of Kakatiyas, Vijayanagara Empire, Sultanate of Golconda, Mughal Empire and Arcot Nawabdom. In the 18th century, Nellore was taken over by the British from the Arcot Nawabs and was part of the Madras Presidency of British India."
passage
result=pred.predict(passage=passage,
             question="which river is flowing in nellore")

result['best_span_str']

