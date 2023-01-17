import pandas as pd



def translate():

    # Translate

    toEn = {'data': 'date',

            'stato': 'state',

            'ricoverati_con_sintomi': 'hospitalized_with_symptoms',

            'terapia_intensiva': 'intensive_care',

            'totale_ospedalizzati': 'total_hospitalized',

            'isolamento_domiciliare': 'home_isolation',

            'totale_attualmente_positivi': 'total_currently_positive',

            'nuovi_attualmente_positivi': 'new_currently_positive',

            'dimessi_guariti': 'recovered',

            'deceduti': 'death',

            'totale_casi': 'total_positive_cases',

            'tamponi': 'total_tests',



            'codice_regione' : 'region_code',

            'denominazione_regione' : 'region_denomination',

            'codice_provincia' : 'province_code',

            'denominazione_provincia' : 'province_denomination',

            'sigla_provincia' : 'province_abbreviation',

            'lat' : 'lat',

            'long' : 'long'



    }



    df = pd.read_json('../dati-json/dpc-covid19-ita-andamento-nazionale.json')

    df.columns = [toEn[df.columns[i]] for i in range(len(df.columns))]

    df.to_csv('national_data.csv', index=False, encoding = 'utf-8')



    df = pd.read_json('../dati-json/dpc-covid19-ita-province.json')

    df.columns = [toEn[df.columns[i]] for i in range(len(df.columns))]

    df.to_csv('provincial_data.csv', index=False, encoding = 'utf-8')



    df = pd.read_json('../dati-json/dpc-covid19-ita-regioni.json')

    df.columns = [toEn[df.columns[i]] for i in range(len(df.columns))]

    df.to_csv('regional_data.csv', index=False, encoding = 'utf-8')