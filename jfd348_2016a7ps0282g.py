import numpy as np;

import pandas;

import seaborn;

import math;

from matplotlib import pyplot;

from scipy.cluster.hierarchy import fclusterdata;

from sklearn.cluster import *;



%matplotlib inline
trainData = pandas.read_csv("../input/TRAIN.csv");

testData = pandas.read_csv("../input/TEST.csv");
def transformData(data):

    

    def _rangeToMean(x):

        if (x == '?'):

            return math.nan;

        if (x[0] == '>'):

            return float(x[1:]);

        i1 = x.index('-');

        a = float(x[1:i1]);

        i2 = x.index(')', i1);

        b = float(x[(i1 + 1):i2]);

        return (a + b) * 0.5;

    

    data["age"] = data["age"].apply(_rangeToMean);

    data["weight"] = data["weight"].apply(_rangeToMean);

    

    drugs = [

        "metformin", "repaglinide", "nateglinide", "chlorpropamide", "glimepiride",

        "acetohexamide", "glipizide", "glyburide", "tolbutamide", "pioglitazone",

        "rosiglitazone", "acarbose", "miglitol", "troglitazone", "tolazamide", "examide", 

        "citoglipton", "insulin", "glyburide-metformin", "glipizide-metformin",

        "glimepiride-pioglitazone", "metformin-rosiglitazone", "metformin-pioglitazone"

    ];

    drugLevelMap = {"No": 0, "Down": 1, "Steady": 2, "Up": 3};

    for drug in drugs:

        data[drug] = data[drug].map(drugLevelMap);

    

    data["race"] = data["race"].map({

        "?": 0, "Caucasian": 1, "AfricanAmerican": 2, "Asian": 3, "Hispanic": 4, "Other": 5,

    });

    data["gender"] = data["gender"].map({"Unknown/Invalid": 0, "Male": 1, "Female": 2});

    data["change"] = data["change"].map({"No": 0, "Ch": 1});

    data["diabetesMed"] = data["diabetesMed"].map({"No": 0, "Yes": 1});

    

    data["payer_code"] = data["payer_code"].map({

        "?": 0, "MC": 1, "MD": 2, "HM": 3, "UN": 4, "BC": 5, "SP": 6,

        "CP": 7, "SI": 8, "DM": 9, "CM": 10,"CH": 11, "PO": 12, "WC": 13,

        "OT": 14, "OG": 15, "MP": 16,

    });

    data["A1Cresult"] = data["A1Cresult"].map({

        "None": 0, "Norm": 1, ">7": 2, ">8": 3,

    });

    data["max_glu_serum"] = data["max_glu_serum"].map({

        "None": 0, "Norm": 1, ">200": 2, ">300": 3,

    });

    

    specialityMap = {

        "?": (0, 0),



        "Pediatrics":                          (1, 0),

        "Pediatrics-Pulmonology":              (1, 1),

        "Pediatrics-Endocrinology":            (1, 2),

        "Pediatrics-CriticalCare":             (1, 3),

        "Pediatrics-Hematology-Oncology":      (1, 4),

        "Pediatrics-Neurology":                (1, 5),

        "Pediatrics-EmergencyMedicine":        (1, 6),

        "Pediatrics-InfectiousDiseases":       (1, 7),

        "Pediatrics-AllergyandImmunology":     (1, 8),



        "InternalMedicine":                    (2, 0),



        "Family/GeneralPractice":              (3, 0),



        "Cardiology":                          (4, 0),

        "Cardiology-Pediatric":                (4, 1),



        "Surgery":                             (5, 0),

        "Surgeon":                             (5, 0),

        "Surgery-General":                     (5, 1),

        "Surgery-Neuro":                       (5, 2),

        "Surgery-Colon&Rectal":                (5, 3),

        "Surgery-Plastic":                     (5, 4),

        "Surgery-Thoracic":                    (5, 5),

        "Surgery-PlasticwithinHeadandNeck":    (5, 6),

        "Surgery-Pediatric":                   (5, 7),

        "Surgery-Maxillofacial":               (5, 8),

        "Surgery-Cardiovascular":              (5, 9),

        "Surgery-Vascular":                    (5, 10),

        "Surgery-Cardiovascular/Thoracic":     (5, 11),

        "SurgicalSpecialty":                   (5, 12),



        "Orthopedics":                         (6, 0),

        "Orthopedics-Reconstructive":          (6, 1),



        "Gastroenterology":                    (7, 0),



        "Nephrology":                          (8, 0),



        "Psychiatry":                          (9, 0),

        "Psychiatry-Child/Adolescent":         (9, 1),

        "Psychology":                          (9, 2),

        "Psychiatry-Addictive":                (9, 3),



        "Emergency/Trauma":                    (10, 0),



        "Pulmonology":                         (11, 0),



        "ObstetricsandGynecology":              (12, 0),

        "Obsterics&Gynecology-GynecologicOnco": (12, 1),

        "Gynecology":                           (12, 2),

        "Obstetrics":                           (12, 3),



        "Hematology/Oncology":                 (13, 0),

        "Otolaryngology":                      (14, 0),

        "Endocrinology":                       (15, 0),

        "Endocrinology-Metabolism":            (15, 1),

        "Urology":                             (16, 0),

        "Neurology":                           (17, 0),



        "Anesthesiology":                      (18, 0),

        "Anesthesiology-Pediatric":            (18, 1),



        "Radiology":                           (19, 0), 

        "Radiologist":                         (19, 0),



        "Podiatry":                            (20, 0),

        "Oncology":                            (21, 0),

        "Ophthalmology":                       (22, 0),

        "PhysicalMedicineandRehabilitation":   (23, 0),

        "InfectiousDiseases":                  (24, 0),

        "Rheumatology":                        (25, 0),

        "AllergyandImmunology":                (26, 0),

        "Dentistry":                           (27, 0),

        "Osteopath":                           (28, 0),

        "PhysicianNotFound":                   (29, 0),

        "Hematology":                          (30, 0),

        "Proctology":                          (31, 0),

        "Pathology":                           (32, 0),

        "Dermatology":                         (33, 0),

        "SportsMedicine":                      (34, 0),

        "Speech":                              (35, 0),

        "Hospitalist":                         (36, 0),

        "OutreachServices":                    (37, 0),

        "Perinatology":                        (38, 0),

        "Neurophysiology":                     (39, 0),

        "DCPTEAM":                             (40, 0),

        "Resident":                            (41, 0),

    };

    

    data["medical_specialty1"] = data["medical_specialty"].apply(lambda x: specialityMap[x][0]);

    data["medical_specialty2"] = data["medical_specialty"].apply(lambda x: specialityMap[x][1]);

    data.drop("medical_specialty", axis = 1, inplace = True);

    

    def _diagType(x):

        if (x[0] == 'V'):

            return 2;

        if (x[0] == 'E'):

            return 3;

        if (x == '?'):

            return 0;

        return 1;

    

    def _diagValue(x):

        if (x[0] == 'V' or x[0] == 'E'):

            return float(x[1:]);

        if (x == '?'):

            return 0;

        return float(x);

    

    data["diag1_type"] = data["diag_1"].apply(_diagType);

    data["diag1_value"] = data["diag_1"].apply(_diagValue);

    data["diag2_type"] = data["diag_2"].apply(_diagType);

    data["diag2_value"] = data["diag_2"].apply(_diagValue);

    data["diag3_type"] = data["diag_3"].apply(_diagType);

    data["diag3_value"] = data["diag_3"].apply(_diagValue);

    data.drop(["diag_1", "diag_2", "diag_3"], axis = 1, inplace = True);

    

    numericalColumns = [

        "age", "weight", "time_in_hospital", "num_lab_procedures", "num_procedures",

        "num_medications", "number_outpatient", "number_emergency", "number_inpatient",

        "number_diagnoses",

    ];

    for col in numericalColumns:

        data[col] = data[col].apply(lambda x: math.nan if (x == '?') else x);

        data[col] = data[col].fillna(data[col].mean());

    

    return data;



transformData(trainData);

transformData(testData);
# Features as (name, type, weight).



featuresList = [

    #("race", 0, 1),

    ("gender", 0, 1),

    ("age", 1, 0.02),

    ("weight", 1, 0.01),

    #("admission_type_id", 0, 1),

    #("discharge_disposition_id", 0, 1),

    #("admission_source_id", 0, 1),

    ("time_in_hospital", 1, 0.1),

    #("payer_code", 0, 1),

    #("medical_specialty1", 2, 1),

    #("medical_specialty2", 2, 1),

    ("num_lab_procedures", 1, 0.005),

    ("num_procedures", 1, 0.1),

    ("num_medications", 1, 0.01),

    ("number_diagnoses", 1, 0.1),

    ("number_outpatient", 1, 0.01),

    ("number_emergency", 1, 0.01),

    ("number_inpatient", 1, 0.01), 

    #("diag1_type", 0, 1),

    #("diag1_value", 1, 0.0005),

    #("diag2_type", 0, 1),

    #("diag2_value", 1, 0.0005),

    #("diag3_type", 0, 1),

    #("diag3_value", 1, 0.0005),

    ("max_glu_serum", 0, 1),

    ("A1Cresult", 0, 1),

    ("metformin", 1, 0.1),

    ("repaglinide", 1, 0.1),

    ("nateglinide", 1, 0.1),

    ("chlorpropamide", 1, 0.1),

    ("glimepiride", 1, 0.1),

    ("acetohexamide", 1, 0.1),

    ("glipizide", 1, 0.1),

    ("glyburide", 1, 0.1),

    ("tolbutamide", 1, 0.1),

    ("pioglitazone", 1, 0.1),

    ("rosiglitazone", 1, 0.1),

    ("acarbose", 1, 0.1),

    ("miglitol", 1, 0.1), 

    ("troglitazone", 1, 0.1),

    ("tolazamide", 1, 0.1),

    ("insulin", 1, 0.1),

    ("glyburide-metformin", 1, 0.1),

    ("glipizide-metformin", 1, 0.1),

    ("glimepiride-pioglitazone", 1, 0.1),

    ("metformin-rosiglitazone", 1, 0.1),

    ("metformin-pioglitazone", 1, 0.1),

    ("change", 0, 1),

    ("diabetesMed", 0, 1)

];



numFeatures = len(featuresList);

featureNames = [x[0] for x in featuresList];

featureTypes = np.array([x[1] for x in featuresList]);

featureWeights = np.array([x[2] for x in featuresList]);
dataX = np.array(trainData[featureNames]) * featureWeights;

cls = KMeans(n_clusters = 100);

cls.fit(dataX);
i = np.min(cls.labels_);

j = np.max(cls.labels_);

clusterTargets = {};

targetVar = trainData['readmitted_NO'];



while (i <= j):

    t = targetVar[cls.labels_ == i];

    s = np.sum(t);

    if (s >= len(t) * 0.5):

        clusterTargets[i] = 1;

    else:

        clusterTargets[i] = 0;

    i += 1;
testX = testData[featureNames];

pred = cls.predict(testX);



with open("result.csv", "w") as f:

    f.write("index,target\n");

    for i in range(0, testX.shape[0]):

        f.write("%d,%d\n" % (testData['index'][i], clusterTargets[pred[i]]));