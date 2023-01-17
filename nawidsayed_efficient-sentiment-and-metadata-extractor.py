def extract_additional_features(pet_id, mode='train'):

    

    sentiment_filename = f'../input/petfinder-adoption-prediction/{mode}_sentiment/{pet_id}.json'

    try:

        sentiment_file = pet_parser.open_json_file(sentiment_filename)

        df_sentiment = pet_parser.parse_sentiment_file(sentiment_file)

        df_sentiment['PetID'] = pet_id

    except FileNotFoundError:

        df_sentiment = []



    dfs_metadata = []

    for ind in range(1,200):

        metadata_filename = '../input/petfinder-adoption-prediction/{}_metadata/{}-{}.json'.format(mode, pet_id, ind)

        try:

            metadata_file = pet_parser.open_json_file(metadata_filename)

            df_metadata = pet_parser.parse_metadata_file(metadata_file)

            df_metadata['PetID'] = pet_id

            dfs_metadata.append(df_metadata)

        except FileNotFoundError:

            break

    if dfs_metadata:

        dfs_metadata = pd.concat(dfs_metadata, ignore_index=True, sort=False)

    dfs = [df_sentiment, dfs_metadata]

    return dfs