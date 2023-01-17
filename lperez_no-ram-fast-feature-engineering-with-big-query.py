# Defines the new features

queries_config = [    
    #########################
    # COUNT
    #########################
    {
        'type': 'count',
        'groupby': ['ip', 'HOUR']   # the HOUR keyword is processed for truncating HOUR from click_time timestamp
    },
    {
        'type': 'count',
        'groupby': ['ip', 'app']
    },
    {
        'type': 'count',
        'groupby': ['ip', 'app', 'os']
    },    
    
    ##########################
    # COUNT UNIQUE
    ##########################
    {
        'type': 'countunique',
        'groupby': ['ip'],
        'unique': 'channel'
    },
    {
        'type': 'countunique',
        'groupby': ['ip', 'device', 'os'],
        'unique': 'app'
    }, 
    {
        'type': 'countunique',
        'groupby': ['ip'],
        'unique': 'app'
    }, 
    {
        'type': 'countunique',
        'groupby': ['ip', 'app'],
        'unique': 'os'
    }, 
    {
        'type': 'countunique',
        'groupby': ['ip'],
        'unique': 'device'
    }, 
    {
        'type': 'countunique',
        'groupby': ['app'],
        'unique': 'channel'
    },   
    
    #######################
    # CUMULATIVE COUNT
    #######################  
    {
        'type': 'cumcount',
        'groupby': ['ip'],
    },   
    {
        'type': 'cumcount',
        'groupby': ['ip', 'device', 'os'],
    },   
    
    #######################
    # NEXT CLICK
    #######################    
    {
        'type': 'last_click',
        'groupby': ['ip', 'app', 'device', 'os', 'channel'],
        'order': 'DESC'       # ASC for last_click, DESC for next_click
    },
    {
        'type': 'last_click',
        'groupby': ['ip', 'os', 'device'],
        'order': 'DESC'       # ASC for last_click, DESC for next_click
    },
    {
        'type': 'last_click',
        'groupby': ['ip', 'os', 'device', 'app'],
        'order': 'DESC'       # ASC for last_click, DESC for next_click
    },
    {
        'type': 'last_click',
        'groupby': ['ip', 'channel'],
        'order': 'DESC'       # ASC for last_click, DESC for next_click
    }, 
    #######################
    # LAST CLICK    
    #######################    
    {
        'type': 'last_click',
        'groupby': ['ip', 'app', 'device', 'os', 'channel'],
        'order': 'ASC'       # ASC for last_click, DESC for next_click
    },
    {
        'type': 'last_click',
        'groupby': ['ip', 'os', 'device'],
        'order': 'ASC'       # ASC for last_click, DESC for next_click
    },
    {
        'type': 'last_click',
        'groupby': ['ip', 'os', 'device', 'app'],
        'order': 'ASC'       # ASC for last_click, DESC for next_click
    },
    {
        'type': 'last_click',
        'groupby': ['ip', 'channel'],
        'order': 'ASC'       # ASC for last_click, DESC for next_click
    },    

    
]
def create_query(main_table_name, train=False):

    query = "#standardSQL \nWITH "
    #####################
    # Create WITH section
    #####################
    with_sections = []
    field_names = []
    temp_table_names = []
    where_clauses = []

    for c in queries_config:
        section = ""
        # Create field_name and table_name
        if c['type'] == 'countunique':
            field_name = c['type'] + "_" + c['unique'] + "_" + "by" + "_" + "_".join(c['groupby'])
        else:
            field_name = c['type'] + "_" + "by" + "_" + "_".join(c['groupby'])
        if 'order' in c:
            field_name += "_" + c['order']
        temp_table_name = field_name + "_table"
        field_names.append(field_name)
        temp_table_names.append(temp_table_name)
        section += temp_table_name + " AS (\n"
        # SELECT
        section += "  SELECT "
        # Insert function to select hours from timestamp when needed
        processed_groupby = [gb if gb != "HOUR" else "TIMESTAMP_TRUNC(click_time, HOUR, 'UTC') as HOUR" for gb in c['groupby']]    
        if   c['type'] == 'count':
            section += ", ".join(processed_groupby) + ", "
            section += "COUNT(*) "
        elif c['type'] == 'countunique':
            section += ", ".join(processed_groupby) + ", "
            section += "COUNT(DISTINCT " + c['unique'] + ") "
        elif c['type'] == 'cumcount':
            section += "index, ROW_NUMBER() OVER (PARTITION BY " + ", ".join(c['groupby']) + " ORDER BY click_time) "
        elif c['type'] == 'last_click':
            section += "index, TIMESTAMP_DIFF(click_time, LAG(click_time) OVER (PARTITION BY " + ", ".join(c['groupby']) + " ORDER BY click_time " + c['order'] + " ), SECOND)\n    "
        section += "as " + field_name + "\n"
        # FROM
        section += "  FROM " + main_table_name + "\n"
        # GROUP BY
        if c['type'] == 'count' or c['type'] == 'countunique' :
            section += "  GROUP BY " + ", ".join(c['groupby']) + "\n"
            where_clause = " AND ".join([main_table_name + "." + gb + " = " + temp_table_name + "." + gb for gb in c['groupby'] ])
            # Process HOUR
            where_clause = where_clause.replace(main_table_name + ".HOUR", "TIMESTAMP_TRUNC(" + main_table_name  +".click_time, HOUR, 'UTC')")
        else:
            where_clause = main_table_name + ".index = " + temp_table_name + ".index"
        section += ")"
        # Append to with_sections
        with_sections.append(section)
        where_clauses.append(where_clause)

    query += ", \n".join(with_sections) + "\n\n"

    #######################
    # Create SELECT section
    #######################
    query += "SELECT\n  "
    if not train:
        query += main_table_name + ".click_id, "
    query += main_table_name + ".ip, " + main_table_name + ".app, " + main_table_name + ".device, " + main_table_name + ".os, " + main_table_name + ".channel, " + main_table_name + ".click_time, "
    if train:
        query += "is_attributed, "
    query += ", ".join(field_names) + "\n"

    #######################
    # Create FROM section
    #######################
    query += "FROM " + main_table_name + ", "
    query += ", ".join(temp_table_names) + "\n"

    #######################
    # Create WHERE section
    #######################

    query += "WHERE\n  "
    query += "\n  AND ".join(where_clauses)
    query += "\n"

    #########################
    # Create ORDER BY section
    #########################

    # query += "ORDER BY ip, click_time" # Not needed for final computation, use it for debug if you wish

    return query
# Create query for train data

train = True
table_name = "`my_dataset.my_table_with_index`"

print(create_query(table_name, train=train))
