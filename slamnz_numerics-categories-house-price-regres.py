from pandas import read_csv, DataFrame



data = read_csv("../input/train.csv")
features = data.columns.tolist()



if "Id" in features:

    features.remove("Id")



print(features)



print("\n Number of Features: %s" % len(data.columns))
features

features_by_dtype = {}



for feature in features:

    

    feature_dtype = str(data.dtypes[feature])

    

    try:

        features_by_dtype[feature_dtype]

    except KeyError:

        features_by_dtype[feature_dtype] = []

        

    

    features_by_dtype[feature_dtype].append(feature)



dtypes = features_by_dtype.keys()



for dtype in dtypes:

    

    count = len(features_by_dtype[dtype])

    print("%s : %s/%s"% (dtype, count, len(features)))

    print(features_by_dtype[dtype])

    print()
categorical_features = features_by_dtype["object"]
feature_description = {

    

    # Building Classification

    

    "MSSubClass" : "The Building Class",

    "MSZoning" : "The General Zoning Classification",

    

    # Target

    "SalePrice" : "Sale Price",

    

    # House Characteristics

    

    "LotFrontage" : "Linear feet of street connected to property",

    "LotArea" : "Lot size in square feet",

    "Street" : "Type of Road Access",

    "Alley" : "Type of Alley Access",

    "LotShape" : "General shape of property",

    "LandContour" : "Flatness of The Property",

    "Utilities" : "Type of utilitists available",

    "LotConfig" : "Lot configuration",

    "LandSlope" : "Slope of property",

    "Neighborhood" : "Physical locations within Ames city limits",

    "Condition1" : "Proximity to main road or railroad",

    "Condition2" : "Proximity to main road or railroad (if a second is present)",

    "BldgType" : "Type of dwelling",

    "HouseStyle" : "Style of dwelling",

    

    # Quality 

    

    "PoolQC" : "Pool quality",

    "Fence" : "Fence quality",

    

    # Rating Variables

    

    "OverallQual" : "Overall material and finish quality (1 - 10 Likert scale)",

    "OverallCond" : "Overall condition rating (1 - 9 Likert scale)",

    

    "RoofStyle" : "Type of roof",

    "RoofMatl" : "Roof material",

    "Exterior1st" : "Exterior covering on house",

    "Exterior2nd" : "Exterior covering on house (if more than one material)",

    

    "MasVnrType" : "Masonry veneer type",

    "MasVnrArea" : "Masonry veneer area in square feet",

    

    "ExterQual" : "Exterior material quality",

    "ExterCond" : "Present condition of the material on the exterior",

    "Foundation" : "Type of foundation",

    

    "BsmtQual" : "Height of the basement",

    "BsmtCond" : "General condition of the basement",

    "BsmtExposure" : "Walkout or garden level basement walls",

    "BsmtFinType1" : "Quality of basement finished area",

    "BsmtFinSF1" : "Type 1 finished square feet",

    "BsmtFinType2" : "Quality of second finished area (if present)",

    "BsmtFinSF2" : "Type 2 finished square feet",

    "BsmtUnfSF" : "Unfinished square feet of basement area",

    "TotalBsmtSF" : "Total square feet of basement area",

    "BsmtFullBath" : "Basement full bathrooms",

    "BsmtHalfBath" : "Basement half bathrooms",

    

    "Heating" : "Type of heating",

    "HeatingQC" : "Heating quality and condition",

    

    "CentralAir" : "Central air conditioning",

    

    "Electrical" : "Electrical system",

    

    "1stFlrSF" : "First Floor square feet",

    "2ndFlrSF" : "Second floor square feet",

    "LowQualFinSF" : "Low quality finished square feet (all floors)",

    

    "GrLivArea" : "Above grade (ground) living area square feet",

    

    "FullBath" : "Full bathrooms above grade",

    "HalfBath" : "Half baths above grade",

    

    "Bedroom" : "Number of bedrooms above basement level",

    

    "Kitchen" : "Number of kitchens",

    "KitchenQual" : "Kitchen quality",

    

    "TotRmsAbvGrd" : "Total rooms above grade (does not include bathrooms)",

    

    "Functional" : "Home functionality rating",

    

    # Fireplace

    

    "Fireplaces" : "Number of fireplaces",

    "FireplaceQu" : "Fireplace quality",

    

    # Garage

    

    "GarageType" : "Garage location",

    "GarageYrBlt" : "Year garage was built",

    "GarageFinish" : "Interior finish of the garage",

    "GarageCars" : "Size of garage in car capacity",

    "GarageArea" : "Size of garage in square feet",

    "GarageQual" : "Garage quality",

    "GarageCond" : "Garage condition",

    

    # Spatial 

    

    "WoodDeckSF" : "Wood deck area in square feet",

    "OpenPorchSF" : "Open porch area in square feet",

    "EnclosedPorch" : "Enclosed porch area in square feet",

    "3SsnPorch" : "Three season porch area in square feet",

    "ScreenPorch" : "Screen porch area in square feet",

    "PoolArea" : "Pool area in square feet",

    

    # Miscellaneous 

    

    "MiscFeature" : "Miscellaneous feature not covered in other categories",

    "MiscVal" : "Value of miscellaneous feature ($)",

    

    # Temporal Characteristics

    

    "MoSold" : "Month Sold",

    "YrSold" : "Year Sold",

    "YearBuilt" : "Original construction date",

    "YearRemodAdd" : "Remodel date",

    

    # Sale Characteristics

    

    "SaleType" : "Type of sale",

    "SaleCondition" : "Condition of sale"

}
for dtype in dtypes:

    

    if dtype == "object":

        continue

    

    print(dtype + "\n")

    

    for feature in features_by_dtype[dtype]:

        

        try:

            print(feature + " : " + feature_description[feature])

        except:

            print(feature + " : " + feature)

            

    print()
float_features = features_by_dtype["float64"]

int_features = features_by_dtype["int64"]

numerical_features = float_features + int_features

remove_list = ["GarageYrBlt", "YearBuilt", "YearRemodAdd", "MoSold", "YrSold", "MSSubClass"]

numerical_features = [n for n in numerical_features if n not in remove_list]
def print_feature_type(feature_list):

    count = len(feature_list)

    print(feature_list)

    print("\n Number of Features: %s" % count)
print_feature_type(numerical_features)
categorical_features = categorical_features + ["MSSubClass"]

print_feature_type(categorical_features)
temporal_features = remove_list.copy()

temporal_features.remove("MSSubClass")



print_feature_type(temporal_features)