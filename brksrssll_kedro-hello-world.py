!kedro info  # requires Kaggle Settings -> Packages Install kedro

!pwd
# requires Kaggle Settings -> Internet On

# used to display a tree of directory structure created by kedro

!apt-get -qq install tree
# Kaggle does not support interactive shell, so creating structure using yaml config instead

import yaml



yaml_dict = dict(output_dir='.',

    project_name='Getting Started',

    repo_name='getting-started',

    python_package='getting_started',

    include_example=True

    )



with open('kedro_new_config.yml', 'w') as outfile:

    yaml.dump(yaml_dict, outfile, default_flow_style=False)



!cat kedro_new_config.yml
!kedro new --config kedro_new_config.yml
%cd ./getting-started

!tree
!kedro test
!kedro run